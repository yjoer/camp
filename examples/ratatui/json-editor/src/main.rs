use std::collections::HashMap;
use std::error::Error;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

fn main() -> Result<(), Box<dyn Error>> {
	let mut terminal = ratatui::init();
	let mut app = App::new();
	let res = run_app(&mut terminal, &mut app);
	ratatui::restore();

	match res {
		Ok(print) => {
			if print {
				app.print_json()?;
			}
		}
		Err(err) => println!("{err:?}"),
	}

	Ok(())
}

fn run_app(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<bool> {
	loop {
		terminal.draw(|frame| ui(frame, app))?;
		let action = handle_events(app)?;

		match action {
			EventAction::Print => return Ok(true),
			EventAction::Skip => {}
			EventAction::Exit => return Ok(false),
		}
	}
}

fn handle_events(app: &mut App) -> std::io::Result<EventAction> {
	if let Event::Key(key) = event::read()? {
		if key.kind == KeyEventKind::Release {
			return Ok(EventAction::Skip);
		}

		match app.current_screen {
			CurrentScreen::Main => match key.code {
				KeyCode::Char('e') => {
					app.current_screen = CurrentScreen::Editing;
					app.currently_editing = Some(CurrentlyEditing::Key);
				}
				KeyCode::Char('q') => {
					app.current_screen = CurrentScreen::Exiting;
				}
				_ => {}
			},
			CurrentScreen::Exiting => match key.code {
				KeyCode::Char('y') => {
					return Ok(EventAction::Print);
				}
				KeyCode::Char('n') | KeyCode::Char('q') => {
					return Ok(EventAction::Exit);
				}
				_ => {}
			},
			CurrentScreen::Editing => match key.code {
				KeyCode::Enter => {
					if let Some(editing) = &app.currently_editing {
						match editing {
							CurrentlyEditing::Key => {
								app.currently_editing = Some(CurrentlyEditing::Value);
							}
							CurrentlyEditing::Value => {
								app.save_key_values();
								app.current_screen = CurrentScreen::Main;
							}
						}
					}
				}
				KeyCode::Backspace => {
					if let Some(editing) = &app.currently_editing {
						match editing {
							CurrentlyEditing::Key => {
								app.key_input.pop();
							}
							CurrentlyEditing::Value => {
								app.value_input.pop();
							}
						}
					}
				}
				KeyCode::Esc => {
					app.current_screen = CurrentScreen::Main;
					app.currently_editing = None;
				}
				KeyCode::Tab => {
					app.toggle_editing();
				}
				KeyCode::Char(value) => {
					if let Some(editing) = &app.currently_editing {
						match editing {
							CurrentlyEditing::Key => {
								app.key_input.push(value);
							}
							CurrentlyEditing::Value => {
								app.value_input.push(value);
							}
						}
					}
				}
				_ => {}
			},
		}
	}

	Ok(EventAction::Skip)
}

enum CurrentScreen {
	Main,
	Editing,
	Exiting,
}

enum CurrentlyEditing {
	Key,
	Value,
}

enum EventAction {
	Print,
	Skip,
	Exit,
}

struct App {
	key_input: String,
	value_input: String,
	pairs: HashMap<String, String>,
	current_screen: CurrentScreen,
	currently_editing: Option<CurrentlyEditing>,
}

impl App {
	pub fn new() -> Self {
		App {
			key_input: String::new(),
			value_input: String::new(),
			pairs: HashMap::new(),
			current_screen: CurrentScreen::Main,
			currently_editing: None,
		}
	}

	fn toggle_editing(&mut self) {
		if let Some(edit_mode) = &self.currently_editing {
			match edit_mode {
				CurrentlyEditing::Key => self.currently_editing = Some(CurrentlyEditing::Value),
				CurrentlyEditing::Value => self.currently_editing = Some(CurrentlyEditing::Key),
			}
		} else {
			self.currently_editing = Some(CurrentlyEditing::Key);
		}
	}

	fn save_key_values(&mut self) {
		let k = self.key_input.clone();
		let v = self.value_input.clone();
		self.pairs.insert(k, v);

		self.key_input.clear();
		self.value_input.clear();
		self.currently_editing = None;
	}

	fn print_json(&self) -> serde_json::Result<()> {
		let output = serde_json::to_string(&self.pairs)?;
		println!("{}", output);
		Ok(())
	}
}

fn ui(frame: &mut Frame, app: &App) {
	let chunks = Layout::default()
		.direction(Direction::Vertical)
		.constraints([
			Constraint::Length(3),
			Constraint::Min(1),
			Constraint::Length(3),
		])
		.split(frame.area());

	let title_text = Text::styled("Create New JSON", Style::default().fg(Color::Green));
	let title_block = Block::default()
		.borders(Borders::ALL)
		.style(Style::default());
	let title = Paragraph::new(title_text).block(title_block);

	frame.render_widget(title, chunks[0]);

	let mut list_items = Vec::<ListItem>::new();
	for key in app.pairs.keys() {
		list_items.push(ListItem::new(Line::from(Span::styled(
			format!("{}: {}", key, app.pairs.get(key).unwrap()),
			Style::default().fg(Color::Yellow),
		))));
	}

	let list = List::new(list_items);
	frame.render_widget(list, chunks[1]);

	let current_navigation_text = vec![
		match app.current_screen {
			CurrentScreen::Main => Span::styled("Normal Mode", Style::default().fg(Color::Green)),
			CurrentScreen::Editing => {
				Span::styled("Editing Mode", Style::default().fg(Color::Yellow))
			}
			CurrentScreen::Exiting => Span::styled("Exiting", Style::default().fg(Color::LightRed)),
		}
		.to_owned(),
		Span::styled(" | ", Style::default().fg(Color::White)),
		{
			if let Some(editing) = &app.currently_editing {
				match editing {
					CurrentlyEditing::Key => {
						Span::styled("Editing JSON Key", Style::default().fg(Color::Green))
					}
					CurrentlyEditing::Value => {
						Span::styled("Editing JSON Value", Style::default().fg(Color::LightGreen))
					}
				}
			} else {
				Span::styled("Not Editing Anything", Style::default().fg(Color::DarkGray))
			}
		},
	];

	let mode_footer = Paragraph::new(Line::from(current_navigation_text))
		.block(Block::default().borders(Borders::ALL));

	let current_key_hint = {
		match app.current_screen {
			CurrentScreen::Main => Span::styled(
				"(q) to quit / (e) to create a new pair",
				Style::default().fg(Color::LightBlue),
			),
			CurrentScreen::Editing => Span::styled(
				"(Esc) to cancel / (Tab) to switch boxes / (Enter) to complete",
				Style::default().fg(Color::LightBlue),
			),
			CurrentScreen::Exiting => Span::from(""),
		}
	};

	let keynotes_footer =
		Paragraph::new(Line::from(current_key_hint)).block(Block::default().borders(Borders::ALL));

	let footer_chunk = Layout::default()
		.direction(Direction::Horizontal)
		.constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
		.split(chunks[2]);

	frame.render_widget(mode_footer, footer_chunk[0]);
	frame.render_widget(keynotes_footer, footer_chunk[1]);

	if let Some(editing) = &app.currently_editing {
		let popup_block = Block::default()
			.title("Enter a new key-value pair")
			.borders(Borders::NONE)
			.style(Style::default().bg(Color::DarkGray));

		let area = centered_rect(60, 25, frame.area());
		frame.render_widget(popup_block, area);

		let popup_chunk = Layout::default()
			.direction(Direction::Horizontal)
			.margin(1)
			.constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
			.split(area);

		let mut key_block = Block::default().title("Key").borders(Borders::ALL);
		let mut value_block = Block::default().title("Value").borders(Borders::ALL);

		let active_style = Style::default().bg(Color::LightYellow).fg(Color::Black);
		match editing {
			CurrentlyEditing::Key => key_block = key_block.style(active_style),
			CurrentlyEditing::Value => value_block = value_block.style(active_style),
		}

		let key_text = Paragraph::new(app.key_input.clone()).block(key_block);
		let value_text = Paragraph::new(app.value_input.clone()).block(value_block);

		frame.render_widget(key_text, popup_chunk[0]);
		frame.render_widget(value_text, popup_chunk[1]);
	}

	if let CurrentScreen::Exiting = app.current_screen {
		frame.render_widget(Clear, chunks[1]);

		let popup_block = Block::default()
			.title("Exiting")
			.borders(Borders::NONE)
			.style(Style::default().bg(Color::DarkGray));

		let exit_paragraph = Paragraph::new(Text::styled(
			"Would you like to output the buffer as JSON? (y/n)",
			Style::default().fg(Color::Gray),
		))
		.block(popup_block)
		.wrap(Wrap { trim: false });

		let area = centered_rect(60, 25, frame.area());
		frame.render_widget(exit_paragraph, area);
	}
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
	let popup_layout = Layout::default()
		.direction(Direction::Vertical)
		.constraints(
			[
				Constraint::Percentage((100 - percent_y) / 2),
				Constraint::Percentage(percent_y),
				Constraint::Percentage((100 - percent_y) / 2),
			]
			.as_ref(),
		)
		.split(r);

	Layout::default()
		.direction(Direction::Horizontal)
		.constraints(
			[
				Constraint::Percentage((100 - percent_x) / 2),
				Constraint::Percentage(percent_x),
				Constraint::Percentage((100 - percent_x) / 2),
			]
			.as_ref(),
		)
		.split(popup_layout[1])[1]
}
