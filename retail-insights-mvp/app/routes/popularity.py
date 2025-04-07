from flask import Blueprint, render_template, request

popularity_bp = Blueprint('popularity', __name__)

@popularity_bp.route('/', methods=['GET', 'POST'])
def popularity():
    prediction = None
    if request.method == 'POST':
        # Call your model here
        prediction = "Popular"
    return render_template('popularity.html', prediction=prediction)
