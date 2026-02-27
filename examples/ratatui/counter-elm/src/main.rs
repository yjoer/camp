use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use ratatui::Frame;
use ratatui::widgets::Paragraph;

fn main() -> io::Result<()> {
	let mut terminal = ratatui::init();
	let mut model = Model::default();

	while model.running_state != RunningState::Done {
		terminal.draw(|f| view(&model, f))?;
		let mut msg = handle_events()?;

		while msg.is_some() {
			msg = update(&mut model, msg.unwrap());
		}
	}

	ratatui::restore();
	Ok(())
}

fn handle_events() -> io::Result<Option<Message>> {
	if event::poll(Duration::from_millis(250))? {
		if let Event::Key(key) = event::read()? {
			if key.kind == event::KeyEventKind::Press {
				match key.code {
					KeyCode::Right => return Ok(Some(Message::Increment)),
					KeyCode::Left => return Ok(Some(Message::Decrement)),
					KeyCode::Char('q') => return Ok(Some(Message::Quit)),
					_ => return Ok(None),
				}
			}
		}
	}

	Ok(None)
}

#[derive(Debug, Default)]
struct Model {
	counter: i32,
	running_state: RunningState,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum RunningState {
	#[default]
	Running,
	Done,
}

enum Message {
	Increment,
	Decrement,
	Reset,
	Quit,
}

fn update(model: &mut Model, msg: Message) -> Option<Message> {
	match msg {
		Message::Increment => {
			model.counter += 1;

			if model.counter > 50 {
				return Some(Message::Reset);
			}
		}
		Message::Decrement => {
			model.counter -= 1;

			if model.counter < -50 {
				return Some(Message::Reset);
			}
		}
		Message::Reset => {
			model.counter = 0;
		}
		Message::Quit => {
			model.running_state = RunningState::Done;
		}
	}

	None
}

fn view(model: &Model, frame: &mut Frame) {
	frame.render_widget(
		Paragraph::new(format!("Counter: {}", model.counter)),
		frame.area(),
	);
}
