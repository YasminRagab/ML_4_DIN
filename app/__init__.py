from flask import Flask
from app.routes import main
import base64

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(main)
    
    # Add b64encode filter
    app.jinja_env.filters['b64encode'] = lambda x: base64.b64encode(x).decode()

    return app
