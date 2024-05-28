from flask import Flask
import base64

def create_app():
    app = Flask(__name__)

    from .routes import main
    app.register_blueprint(main)

    app.jinja_env.filters['b64encode'] = lambda x: base64.b64encode(x).decode()

    return app
