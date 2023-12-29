use llm::{
    load, load_progress_callback_stdout, models::Llama, InferenceError, InferenceRequest, Model,
    OutputRequest, TokenUtf8Buffer,
};
use rand::thread_rng;

pub fn get_model(path: &str) -> Llama {
    load::<Llama>(
        std::path::Path::new(path),
        Default::default(),
        load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("failed to load model from path {path}: {err}"))
}

pub fn infer(
    model: &Llama,
    prompt: String,
    callback: impl FnMut(&str) -> Result<(), ()>,
) -> Result<(), String> {
    sesson_infer(
        model,
        &mut thread_rng(),
        &InferenceRequest {
            prompt: prompt.as_str(),
            maximum_token_count: Some(300),
            ..Default::default()
        },
        &mut Default::default(),
        callback,
    )
}

fn sesson_infer(
    model: &Llama,
    rng: &mut impl rand::Rng,
    request: &InferenceRequest,
    output_request: &mut OutputRequest,
    mut callback: impl FnMut(&str) -> Result<(), ()>,
) -> Result<(), String> {
    let mut session = model.start_session(Default::default());
    let maximum_token_count = request.maximum_token_count.unwrap_or(usize::MAX);
    let parameters = request.parameters.unwrap_or(model.inference_parameters());
    session
        .feed_prompt(
            model,
            parameters,
            request.prompt,
            output_request,
            |_| -> Result<(), std::convert::Infallible> { Ok(()) },
        )
        .unwrap();
    let mut tokens_processed = 0;
    let mut token_utf8_buf = TokenUtf8Buffer::new();
    while tokens_processed < maximum_token_count {
        let token = match session.infer_next_token(model, parameters, &mut Default::default(), rng)
        {
            Ok(token) => token,
            Err(InferenceError::EndOfText) => break,
            Err(e) => return Err(e.to_string()),
        };

        if let Some(tokens) = token_utf8_buf.push(token) {
            if let Err(_) = callback(&tokens) {
                break;
            }
        }

        tokens_processed += 1;
    }
    if maximum_token_count == tokens_processed {
        Err("reached maximum token limit".to_string())
    } else {
        Ok(())
    }
}
