from flask import Blueprint, render_template

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
def dashboard():
    # Dummy data for now – replace with real data from your models later
    popular_data = {
        "labels": ["Jan", "Feb", "Mar", "Apr"],
        "values": [30, 45, 60, 40]
    }

    segmentation_data = {
        "labels": ["High Value", "Medium Value", "Low Value"],
        "values": [10, 30, 60]
    }

    user_engagement_data = {
        "labels": ["Visited", "Interacted", "Converted"],
        "values": [80, 40, 20]
    }

    return render_template(
        'dashboard.html',
        popular_data=popular_data,
        segmentation_data=segmentation_data,
        engagement_data=user_engagement_data
    )
