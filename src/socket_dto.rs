use serde::{Deserialize, Serialize};
use websocket::OwnedMessage;

#[derive(Debug, Serialize)]
pub enum SocketResponseLevel {
    #[serde(rename(serialize = "success"))]
    Success,
    #[serde(rename(serialize = "end-of-text"))]
    EndOfText,
    #[serde(rename(serialize = "busy"))]
    Busy,
    #[serde(rename(serialize = "error"))]
    Error,
}

#[derive(Debug, Serialize)]
pub struct SocketResponse<'a> {
    level: SocketResponseLevel,
    message: Option<&'a str>,
}

impl<'a> SocketResponse<'a> {
    pub fn new(level: SocketResponseLevel, message: &'a str) -> Self {
        Self {
            level,
            message: Some(message),
        }
    }

    pub fn end_of_text() -> Self {
        Self {
            level: SocketResponseLevel::EndOfText,
            message: None,
        }
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
pub struct ModelParameters {
    #[serde(rename(deserialize = "top-k"))]
    pub top_k: Option<usize>,
    #[serde(rename(deserialize = "top-p"))]
    pub top_p: Option<f32>,
    #[serde(rename(deserialize = "repeat-penalty"))]
    pub repeat_penalty: Option<f32>,
    pub temperature: Option<f32>,
    #[serde(rename(deserialize = "repetition-penalty-last-n"))]
    pub repetition_penalty_last_n: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SocketRequest {
    pub action: SocketRequestAction,
    pub message: Option<String>,
    pub parameters: Option<ModelParameters>,
}
