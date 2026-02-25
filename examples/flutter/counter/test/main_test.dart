import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_counter/main.dart';

void main() {
  testWidgets('counter', (WidgetTester tester) async {
    await tester.pumpWidget(const MainApp());

    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    await tester.tap(find.text('Increment'));
    await tester.pump();

    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);

    await tester.tap(find.text('Decrement'));
    await tester.pump();

    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);
  });
}
