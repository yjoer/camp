package com.yjoer.samples

import android.os.Bundle
import android.view.ViewGroup.LayoutParams
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

class SavedInstanceStateActivity : AppCompatActivity() {
	var count = 0
	lateinit var text: TextView

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		text = TextView(this)
		text.text = "Count: $count"

		val button = MaterialButton(this)
		button.layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
		button.text = "+"
		button.setPadding(0, 4.dp, 0, 0)
		button.setOnClickListener { text.text = "Count: ${++count}" }

		val layout = LinearLayout(this)
		layout.orientation = LinearLayout.VERTICAL
		layout.layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
		layout.setPadding(8.dp, 4.dp, 8.dp, 4.dp)
		layout.addView(text)
		layout.addView(button)

		val safe_area = LinearLayout(this)
		safe_area.fitsSystemWindows = true
		safe_area.addView(layout)

		setContentView(safe_area)
	}

	override fun onRestoreInstanceState(savedInstanceState: Bundle) {
		count = savedInstanceState.getInt("count")
		text.text = "Count: $count"
	}

	override fun onSaveInstanceState(outState: Bundle) {
		outState.run { putInt("count", count) }

		super.onSaveInstanceState(outState)
	}
}
