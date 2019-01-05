
import flask_app
import flask_base_annotation
import flask_xls_handling
from tools_usetorch import Flatten

if __name__ == '__main__':
    flask_app.socketio.run(flask_app.app, debug=True)
