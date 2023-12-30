use core::time;
use llm::models::Llama;
use llm::InferenceParameters;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use websocket::server::{NoTlsAcceptor, WsServer};
use websocket::OwnedMessage;

use crate::ml::infer;
use crate::socket_dto::{SocketRequest, SocketRequestAction, SocketResponse, SocketResponseLevel};

pub fn handle_socket(server: WsServer<NoTlsAcceptor, TcpListener>, model: Llama) {
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
            let flag = Arc::new(AtomicBool::new(false));

            let thread_sender = Arc::clone(&sender);
            let thread_flag = Arc::clone(&flag);
            thread::spawn(move || loop {
                if thread_flag.load(Ordering::Relaxed) {
                    continue;
                }
                let mut sender = thread_sender.lock().unwrap();
                let message = rx.recv().unwrap();
                if let Err(_) = sender.send_message(&message) {}
                match message {
                    OwnedMessage::Close(_) => break,
                    _ => {}
                }
                thread::sleep(time::Duration::from_millis(500));
            });

            for message in receiver.incoming_messages() {
                let message = message.unwrap();
                let message_sender = Arc::new(tx.clone());

                match message {
                    OwnedMessage::Close(_) => {
                        println!("[{i}] {ip} disconnected");
                        message_sender.send(OwnedMessage::Close(None)).unwrap();
                        flag.store(true, Ordering::Relaxed);
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
                        match req.action {
                            SocketRequestAction::Stop => flag.store(true, Ordering::Relaxed),
                            SocketRequestAction::Evaluate => {
                                flag.store(false, Ordering::Relaxed);
                                let model = Arc::clone(&model);
                                let closure_sender = Arc::clone(&message_sender);
                                let thread_flag = Arc::clone(&flag);
                                thread::spawn(move || {
                                    if let Ok(model) = model.try_lock() {
                                        println!("[{i}] using model");
                                        let eval_message = match req.message {
                                            Some(m) => m,
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
                                        let model_parameters = match req.parameters {
                                            Some(p) => InferenceParameters {
                                                top_k: p.top_k.unwrap_or(40),
                                                top_p: p.top_p.unwrap_or(0.95),
                                                repeat_penalty: p.repeat_penalty.unwrap_or(1.45),
                                                temperature: p.temperature.unwrap_or(0.85),
                                                repetition_penalty_last_n: p
                                                    .repetition_penalty_last_n
                                                    .unwrap_or(512),
                                                ..InferenceParameters::default()
                                            },
                                            None => InferenceParameters::default(),
                                        };
                                        if let Err(err_message) = infer(
                                            &model,
                                            eval_message,
                                            model_parameters,
                                            move |t: &str| {
                                                let closure_sender = Arc::clone(&closure_sender);
                                                let thread_flag = Arc::clone(&thread_flag);
                                                if thread_flag.load(Ordering::Relaxed) {
                                                    return Err(());
                                                }

                                                let message = SocketResponse::new(
                                                    SocketResponseLevel::Success,
                                                    &t,
                                                )
                                                .to_socket_message();
                                                if let Err(_) = closure_sender.send(message) {
                                                    Err(())
                                                } else {
                                                    Ok(())
                                                }
                                            },
                                        ) {
                                            let message = SocketResponse::new(
                                                SocketResponseLevel::Error,
                                                &err_message,
                                            )
                                            .to_socket_message();
                                            message_sender.send(message).unwrap();
                                        }
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
                    _ => {}
                }
            }
            flag.store(true, Ordering::Relaxed);
        });
    }
}
