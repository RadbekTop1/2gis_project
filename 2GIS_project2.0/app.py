from flask import Flask, request, render_template
from config import ASPECT_PROFILES, DISH_PROFILES
from k3 import recommend

app = Flask(__name__)

@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    top = int(request.args.get("top", 10))
    aspects, dishes, results = [], [], []

    if query:
        aspects, dishes, results = recommend(query, top=top)

    # Список кликабельных тегов для главной страницы
    sample_tags = list(ASPECT_PROFILES.keys()) + [
        "бургер", "пицца", "плов", "шашлык", "суши и роллы", "завтраки", "дети и семья"
    ]

    return render_template(
        "index.html",
        query=query,
        aspects=aspects,
        dishes=dishes,
        results=results,
        sample_tags=sample_tags,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)