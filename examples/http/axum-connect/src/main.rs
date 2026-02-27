use axum::Router;
use axum_connect::prelude::*;
use proto::hello::*;

mod proto {
	pub mod hello {
		include!(concat!(env!("OUT_DIR"), "/hello.v1.rs"));
	}
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
	let app = Router::new()
		.rpc(HelloService::hello(hello))
		.rpc(HelloService::hello_name(hello_name));

	let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

	println!("listening on http://{:?}", listener.local_addr().unwrap());
	axum::serve(listener, app).await.unwrap();
}

async fn hello(_request: HelloRequest) -> HelloResponse {
	HelloResponse {
		hello: "world".into(),
	}
}

async fn hello_name(request: HelloNameRequest) -> HelloNameResponse {
	HelloNameResponse {
		hello: request.name,
	}
}
