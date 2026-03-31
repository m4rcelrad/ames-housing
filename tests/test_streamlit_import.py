import pathlib
import py_compile
import unittest


class TestStreamlitAppSmoke(unittest.TestCase):
    def test_streamlit_app_compiles(self):
        app_path = pathlib.Path("streamlit_app.py")
        self.assertTrue(app_path.exists(), "streamlit_app.py should exist")
        py_compile.compile(str(app_path), doraise=True)


if __name__ == "__main__":
    unittest.main()

