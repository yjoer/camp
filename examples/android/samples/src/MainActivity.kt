package com.yjoer.samples

import android.content.Context
import android.content.Intent
import android.content.res.Resources
import android.os.Bundle
import android.view.ViewGroup.LayoutParams
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.toColorInt
import com.google.android.material.listitem.ListItemCardView
import com.google.android.material.listitem.ListItemLayout
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		val layout = LinearLayout(this)
		layout.layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
		layout.orientation = LinearLayout.VERTICAL
		layout.setPadding(8.dp, 4.dp, 8.dp, 4.dp)

		val activity_items: MutableMap<String, () -> Unit> = mutableMapOf()
		activity_items["Saved Instance State"] = {
			startActivity(Intent(this, SavedInstanceStateActivity::class.java))
		}

		val sections: MutableMap<String, Map<String, () -> Unit>> = mutableMapOf()
		sections["Activity"] = activity_items

		for ((section, items) in sections) {
			val t = TextView(this)
			t.text = section
			t.layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
			t.setBackgroundColor("#ffffdd00".toColorInt())
			t.setPadding(16.dp, 1.dp, 16.dp, 1.dp)
			layout.addView(t)

			for ((k, v) in items) {
				val item = ListItem(this, k, v)
				layout.addView(item)
			}
		}

		val safe_area = LinearLayout(this)
		safe_area.fitsSystemWindows = true
		safe_area.addView(layout)

		setContentView(safe_area)
	}
}

val Int.dp: Int
	get() = (this * Resources.getSystem().displayMetrics.density).roundToInt()

val Float.dp: Int
	get() = (this * Resources.getSystem().displayMetrics.density).roundToInt()

fun ListItem(context: Context, text: String, on_click: () -> Unit): ListItemLayout {
	val t = TextView(context)
	t.text = text

	val list_item = LinearLayout(context)
	list_item.orientation = LinearLayout.HORIZONTAL
	list_item.addView(t)

	val card = ListItemCardView(context)
	card.addView(list_item)
	card.isClickable = true
	card.setOnClickListener { on_click() }

	val layout = ListItemLayout(context)
	layout.addView(card)

	return layout
}
