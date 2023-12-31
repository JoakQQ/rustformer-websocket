extern crate rustformer_websocket;

use rustformer_websocket::{ml::get_model, socket::handle_socket};
use websocket::sync::Server;

fn main() {
    // model.bin downloaded from huggingface
    let model_path = "./models/open_llama_3b-f16.bin";

    let llama = get_model::<llm::models::Llama>(model_path);
    let server = Server::bind("127.0.0.1:3001").unwrap();
    handle_socket(server, llama);
}
