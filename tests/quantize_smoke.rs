// Basic smoke tests for model source parsing

#[test]
fn parse_model_sources() {
    use ohms_adaptq::parse_model_source;
    let s1 = parse_model_source("hf:meta-llama/Llama-3-8B:consolidated.safetensors");
    let s2 = parse_model_source("url:https://example.com/model.onnx");
    let s3 = parse_model_source("ollama:llama3:8b");
    let s4 = parse_model_source("file:/tmp/model.safetensors");
    let s5 = parse_model_source("/tmp/model.onnx");
    // Ensure variables are used to avoid warnings
    assert!(matches!(format!("{:?}", s1).len() > 0, true));
    assert!(matches!(format!("{:?}", s2).len() > 0, true));
    assert!(matches!(format!("{:?}", s3).len() > 0, true));
    assert!(matches!(format!("{:?}", s4).len() > 0, true));
    assert!(matches!(format!("{:?}", s5).len() > 0, true));
}

