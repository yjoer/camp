import 'package:flutter/widgets.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return WidgetsApp(
      color: const Color(0xffffffff),
      home: Container(
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        child: Column(children: [Counter()]),
      ),
      pageRouteBuilder: <T>(settings, builder) => PageRouteBuilder(
        pageBuilder: (context, _, _) => builder(context),
        transitionsBuilder: (_, _, _, child) => child,
      ),
    );
  }
}

class Counter extends StatefulWidget {
  const Counter({super.key});

  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  void _decrement() {
    setState(() {
      _count--;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          spacing: 8,
          children: [
            Button('Increment', onTap: _increment),
            Container(
              alignment: Alignment.center,
              constraints: BoxConstraints(minWidth: 48),
              child: Text(_count.toString(), style: TextStyle(fontSize: 16)),
            ),
            Button('Decrement', onTap: _decrement),
          ],
        ),
      ],
    );
  }
}

class Button extends StatefulWidget {
  const Button(this.text, {super.key, required this.onTap});

  final String text;
  final VoidCallback onTap;

  @override
  State<Button> createState() => _ButtonState();
}

class _ButtonState extends State<Button> {
  double _scale = 1.0;

  void _handleTapDown() {
    setState(() {
      _scale = 0.95;
    });
  }

  void _handleTapUp() {
    setState(() {
      _scale = 1.0;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: widget.onTap,
        onTapDown: (_) => _handleTapDown(),
        onTapUp: (_) => _handleTapUp(),
        onTapCancel: () => _handleTapUp(),
        child: AnimatedScale(
          scale: _scale,
          duration: const Duration(milliseconds: 100),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
            decoration: BoxDecoration(
              color: const Color.fromRGBO(0, 0, 0, 0.04),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Container(
              alignment: Alignment.center,
              height: 24,
              child: Text(widget.text, style: TextStyle(fontSize: 16)),
            ),
          ),
        ),
      ),
    );
  }
}
