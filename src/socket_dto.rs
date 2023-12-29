use serde::{Deserialize, Serialize};
use websocket::OwnedMessage;

#[derive(Debug, Serialize)]
pub enum SocketResponseLevel {
    #[serde(rename(serialize = "success"))]
    Success,
    #[serde(rename(serialize = "busy"))]
    Busy,
    #[serde(rename(serialize = "error"))]
    Error,
}

#[derive(Debug, Serialize)]
pub struct SocketResponse<'a> {
    level: SocketResponseLevel,
    message: &'a str,
}

impl<'a> SocketResponse<'a> {
    pub fn new(level: SocketResponseLevel, message: &'a str) -> Self {
        Self { level, message }
    }

    pub fn to_socket_message(&self) -> OwnedMessage {
        OwnedMessage::Text(serde_json::to_string(&self).unwrap())
    }
}

#[derive(Debug, Deserialize)]
pub enum SocketRequestAction {
    #[serde(rename(deserialize = "stop"))]
    Stop,
    #[serde(rename(deserialize = "evaluate"))]
    Evaluate,
}

#[derive(Debug, Deserialize)]
pub struct SocketRequest {
    pub action: SocketRequestAction,
    #[serde(default)]
    pub message: String,
}
