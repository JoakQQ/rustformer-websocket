use llm::{
    load, load_progress_callback_stdout, models::Llama, InferenceError, InferenceParameters,
    InferenceRequest, Model, OutputRequest, TokenUtf8Buffer,
};
use rand::thread_rng;

const NUMBER_OF_THREADS: usize = 4;

pub fn get_model(path: &str) -> Llama {
    load::<Llama>(
        std::path::Path::new(path),
        Default::default(),
        load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("failed to load model from path {path}: {err}"))
}

pub enum StepResult {
    EndOfText,
    Result(String),
}

pub fn infer(
    model: &Llama,
    prompt: String,
    parameters: InferenceParameters,
    callback: impl FnMut(StepResult) -> Result<(), ()>,
) -> Result<(), String> {
    sesson_infer(
        model,
        &mut thread_rng(),
        &InferenceRequest {
            prompt: prompt.as_str(),
            maximum_token_count: Some(300),
            parameters: Some(&InferenceParameters {
                n_threads: NUMBER_OF_THREADS,
                ..parameters
            }),
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
    mut callback: impl FnMut(StepResult) -> Result<(), ()>,
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
            Err(InferenceError::EndOfText) => {
                if let Err(_) = callback(StepResult::EndOfText) {}
                break;
            },
            Err(e) => return Err(e.to_string()),
        };

        if let Some(tokens) = token_utf8_buf.push(token) {
            if let Err(_) = callback(StepResult::Result(tokens)) {
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
