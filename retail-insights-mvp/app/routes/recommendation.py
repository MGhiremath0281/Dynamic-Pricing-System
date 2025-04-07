from flask import Blueprint, render_template, request

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/', methods=['GET', 'POST'])
def recommendation():
    recommendations = None

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        # Call your recommendation model here
        # Example dummy logic:
        recommendations = [f"Product A for User {user_id}", "Product B", "Product C"]

    return render_template('recommendation.html', recommendations=recommendations)
