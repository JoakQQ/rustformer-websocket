use core::time;
use llm::{InferenceParameters, Model};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use websocket::server::{NoTlsAcceptor, WsServer};
use websocket::sync::Writer;
use websocket::OwnedMessage;

use crate::ml::{infer, StepResult};
use crate::socket_dto::{SocketRequest, SocketRequestAction, SocketResponse, SocketResponseLevel};

pub fn handle_socket(server: WsServer<NoTlsAcceptor, TcpListener>, model: Box<dyn Model>) {
    let model = Arc::new(Mutex::new(model));
    for (i, request) in server.filter_map(Result::ok).enumerate() {
        let model = Arc::clone(&model);
        thread::spawn(move || {
            let client = request.accept().unwrap();
            let ip = client.peer_addr().unwrap();
            println!("[{i}] {ip} connected");
            let (mut receiver, sender) = client.split().unwrap();
            let sender = Arc::new(Mutex::new(sender));
            let (tx, rx) = mpsc::channel::<OwnedMessage>();
            let thread_flag = Arc::new(AtomicBool::new(false));

            init_websocket_sender(rx, &thread_flag, &sender);

            for message in receiver.incoming_messages() {
                let message = message.unwrap();
                let message_sender = Arc::new(tx.clone());

                match message {
                    OwnedMessage::Close(_) => {
                        println!("[{i}] {ip} disconnected");
                        message_sender.send(OwnedMessage::Close(None)).unwrap();
                        thread_flag.store(true, Ordering::Relaxed);
                        return;
                    }
                    OwnedMessage::Text(text) => {
                        let req = match serde_json::from_str::<SocketRequest>(&text) {
                            Ok(req) => req,
                            Err(err) => {
                                let message = SocketResponse::new(
                                    SocketResponseLevel::Error,
                                    &err.to_string(),
                                )
                                .to_socket_message();
                                message_sender.send(message).unwrap();
                                continue;
                            }
                        };
                        handle_websocket_text_request(
                            i,
                            req,
                            &model,
                            &thread_flag,
                            &message_sender,
                        );
                    }
                    _ => {}
                }
            }
            thread_flag.store(true, Ordering::Relaxed);
        });
    }
}

fn init_websocket_sender(
    message_receiver: Receiver<OwnedMessage>,
    flag: &Arc<AtomicBool>,
    sender: &Arc<Mutex<Writer<TcpStream>>>,
) {
    let thread_sender = Arc::clone(&sender);
    let thread_flag = Arc::clone(&flag);
    thread::spawn(move || loop {
        if thread_flag.load(Ordering::Relaxed) {
            continue;
        }
        let mut sender = thread_sender.lock().unwrap();
        let message = message_receiver.recv().unwrap();
        if let Err(_) = sender.send_message(&message) {}
        match message {
            OwnedMessage::Close(_) => break,
            _ => {}
        }
        thread::sleep(time::Duration::from_millis(500));
    });
}

fn handle_websocket_text_request(
    current_connection: usize,
    req: SocketRequest,
    model: &Arc<Mutex<Box<dyn Model>>>,
    thread_flag: &Arc<AtomicBool>,
    message_sender: &Arc<Sender<OwnedMessage>>,
) {
    let message_sender = message_sender.clone();
    match req.action {
        SocketRequestAction::Stop => thread_flag.store(true, Ordering::Relaxed),
        SocketRequestAction::Evaluate => {
            thread_flag.store(false, Ordering::Relaxed);
            let model = Arc::clone(model);
            let closure_sender = Arc::clone(&message_sender);
            let thread_flag = Arc::clone(&thread_flag);
            thread::spawn(move || {
                if let Ok(model) = model.try_lock() {
                    println!("[{current_connection}] locked the model");
                    let eval_message = match req.message.as_ref() {
                        Some(m) => m.to_string(),
                        None => {
                            let message = SocketResponse::new(
                                SocketResponseLevel::Error,
                                "the `message` field is missing",
                            )
                            .to_socket_message();
                            message_sender.send(message).unwrap();
                            return;
                        }
                    };
                    if let Err(err_message) = infer(
                        model.as_ref(),
                        eval_message,
                        get_request_model_parameters(&req),
                        move |res| {
                            let thread_flag = Arc::clone(&thread_flag);
                            if thread_flag.load(Ordering::Relaxed) {
                                return Err(());
                            }

                            let closure_sender = Arc::clone(&closure_sender);
                            let message = match res {
                                StepResult::EndOfText => {
                                    SocketResponse::end_of_text().to_socket_message()
                                }
                                StepResult::Result(s) => {
                                    SocketResponse::new(SocketResponseLevel::Success, s.as_str())
                                        .to_socket_message()
                                }
                            };
                            if let Err(_) = closure_sender.send(message) {
                                Err(())
                            } else {
                                Ok(())
                            }
                        },
                    ) {
                        let message = SocketResponse::new(SocketResponseLevel::Error, &err_message)
                            .to_socket_message();
                        message_sender.send(message).unwrap();
                    }
                    println!("[{current_connection}] unlocked the model");
                } else {
                    let message = SocketResponse::new(
                        SocketResponseLevel::Busy,
                        "please wait a second, the server is busy now ...",
                    )
                    .to_socket_message();
                    message_sender.send(message).unwrap();
                    return;
                }
            });
        }
    }
}

fn get_request_model_parameters(req: &SocketRequest) -> InferenceParameters {
    match req.parameters.as_ref() {
        Some(p) => InferenceParameters {
            top_k: p.top_k.unwrap_or(40),
            top_p: p.top_p.unwrap_or(0.95),
            repeat_penalty: p.repeat_penalty.unwrap_or(1.45),
            temperature: p.temperature.unwrap_or(0.85),
            repetition_penalty_last_n: p.repetition_penalty_last_n.unwrap_or(512),
            ..InferenceParameters::default()
        },
        None => InferenceParameters::default(),
    }
}
