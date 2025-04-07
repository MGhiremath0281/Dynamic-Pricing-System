# app/__init__.py

from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'  # Set this securely

    # Register Blueprints
    from app.routes.home import home_bp
    from app.routes.popularity import popularity_bp
    from app.routes.recommendation import recommendation_bp
    from app.routes.segmentation import segmentation_bp
    ## from app.routes.basket import basket_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(popularity_bp, url_prefix='/popularity')
    app.register_blueprint(recommendation_bp, url_prefix='/recommendation')
    app.register_blueprint(segmentation_bp, url_prefix='/segmentation')
   ## app.register_blueprint(basket_bp, url_prefix='/basket')
    from app.routes.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')


    return app
