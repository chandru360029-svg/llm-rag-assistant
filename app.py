import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from rag import run_graph
from tools import estimate_furniture_delivery

load_dotenv()
app = Flask(__name__)


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("Please choose Yes or No for lift and assembly.")


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    tool_result = None
    error = None
    question_value = ""

    form_values = {
        "distance_km": "10",
        "item_count": "1",
        "floor": "0",
        "lift_available": "true",
        "assembly_required": "false",
    }

    if request.method == "POST":
        action = request.form.get("action", "ask")

        if action == "ask":
            question_value = request.form.get("question", "").strip()
            if not question_value:
                error = "Please enter a question."
            else:
                try:
                    result = run_graph({
                        "action": "ask",
                        "question": question_value,
                    })
                    answer = result.get("answer")
                    sources = result.get("sources", [])
                except Exception as exc:
                    app.logger.exception("Document question failed")
                    error = str(exc)

        elif action == "estimate_delivery":
            for key in form_values:
                form_values[key] = request.form.get(key, form_values[key])
            try:
                distance = float(form_values["distance_km"])
                items = int(form_values["item_count"])
                floor = int(form_values["floor"])
                lift = parse_bool(form_values["lift_available"])
                assembly = parse_bool(form_values["assembly_required"])

                result = run_graph({
                    "action": "estimate_delivery",
                    "tool_inputs": {
                        "distance_km": distance,
                        "item_count": items,
                        "floor": floor,
                        "lift_available": lift,
                        "assembly_required": assembly,
                    },
                })
                tool_result = result.get("tool_result")
            except (ValueError, TypeError) as exc:
                error = str(exc)
            except Exception as exc:
                app.logger.exception("Delivery estimate failed")
                error = str(exc)

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        tool_result=tool_result,
        error=error,
        question_value=question_value,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
