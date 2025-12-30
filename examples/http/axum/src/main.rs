use axum::Json;
use axum::Router;
use axum::routing::get;
use serde_json::Value;
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let app = Router::new().route("/", get(hello));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    println!("listening on http://{:?}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn hello() -> Json<Value> {
    Json(json!({"hello": "world"}))
}
