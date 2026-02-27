use std::io::Result;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::symbols::border;
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Paragraph, Widget};
use ratatui::{DefaultTerminal, Frame};

fn main() -> Result<()> {
	let mut terminal = ratatui::init();
	let result = App::default().run(&mut terminal);
	ratatui::restore();
	result
}

#[derive(Debug, Default)]
struct App {
	counter: i32,
	exit: bool,
}

impl App {
	pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
		while !self.exit {
			terminal.draw(|frame| self.draw(frame))?;
			self.handle_events()?;
		}

		Ok(())
	}

	fn draw(&self, frame: &mut Frame) {
		frame.render_widget(self, frame.area());
	}

	fn handle_events(&mut self) -> Result<()> {
		let event = event::read();

		if let Ok(Event::Key(key)) = event {
			let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
			let press = key.kind == KeyEventKind::Press;

			if let KeyCode::Char(c) = key.code {
				let k1 = c.eq_ignore_ascii_case(&'q');
				let k2 = ctrl && c.eq_ignore_ascii_case(&'c');

				if k1 || k2 {
					self.exit = true;
				}
			}

			if press && key.code == KeyCode::Left {
				self.counter -= 1;
			} else if press && key.code == KeyCode::Right {
				self.counter += 1;
			}
		}

		Ok(())
	}
}

impl Widget for &App {
	fn render(self, area: Rect, buf: &mut Buffer) {
		let title = Line::from(" Counter App Example ".bold());
		let instructions = Line::from(vec![
			" Decrement".into(),
			" <Left>".blue().bold(),
			" Increment".into(),
			" <Right>".blue().bold(),
			" Quit".into(),
			" <Q> ".blue().bold(),
		]);
		let block = Block::bordered()
			.title(title.centered())
			.title_bottom(instructions.centered())
			.border_set(border::THICK);

		let counter_text = Text::from(vec![Line::from(vec![
			"Value: ".into(),
			self.counter.to_string().yellow(),
		])]);

		Paragraph::new(counter_text)
			.centered()
			.block(block)
			.render(area, buf);
	}
}
