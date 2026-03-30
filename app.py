from gradio_case_app import build_demo, start_background_prewarm


if __name__ == "__main__":
    start_background_prewarm()
    build_demo().launch()
