use core::time;
use llm::models::Llama;
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
            thread::spawn(move || loop {
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
                                thread::spawn(move || loop {
                                    if let Ok(model) = model.try_lock() {
                                        println!("[{i}] using model");
                                        if let Err(err_message) =
                                            infer(&model, req.message, move |t: &str| {
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
                                            })
                                        {
                                            let message = SocketResponse::new(
                                                SocketResponseLevel::Error,
                                                &err_message,
                                            )
                                            .to_socket_message();
                                            message_sender.send(message).unwrap();
                                        }
                                        break;
                                    } else {
                                        thread::sleep(time::Duration::from_secs(1));
                                        let message = SocketResponse::new(
                                            SocketResponseLevel::Busy,
                                            "please wait a second, the server is busy now ...",
                                        )
                                        .to_socket_message();
                                        message_sender.send(message).unwrap();
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
