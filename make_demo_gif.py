"""Render a small scripted animation of real findex query results for README."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from findex.model import load_model
from findex.numpy_rank import ranked_search_numpy


def main() -> None:
    model = load_model("index.bin")
    try:
        queries = ["python", "індекс", "python OR база OR тестування"]
        output: list[Image.Image] = []
        font_regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        heading = ImageFont.truetype(font_bold, 48)
        body = ImageFont.truetype(font_regular, 26)
        mini = ImageFont.truetype(font_regular, 19)
        title = ImageFont.truetype(font_bold, 27)
        for query in queries:
            frame = Image.new("RGB", (1100, 620), "#0b1224")
            painter = ImageDraw.Draw(frame)
            painter.rounded_rectangle((55, 48, 1045, 580), 30, fill="#142448", outline="#435a8b", width=2)
            painter.text((95, 75), "findex / search", font=heading, fill="#a3b9ff")
            painter.text((95, 155), "Python · FastAPI · NumPy BM25", font=mini, fill="#a4b9dd")
            painter.rounded_rectangle((95, 205, 1000, 280), 16, fill="#243859", outline="#7992bf")
            painter.text((120, 223), query, font=body, fill="#ffffff")
            painter.rounded_rectangle((830, 216, 985, 267), 12, fill="#7597ff")
            painter.text((854, 229), "Search", font=mini, fill="#06132d")
            found = ranked_search_numpy(model, query, top_k=3)
            painter.text((95, 305), f"Results · {len(model.cached_ids(query))} documents", font=mini, fill="#a0aed1")
            for position, hit in enumerate(found):
                y = 350 + position * 70
                painter.text((95, y), hit.title, font=title, fill="#b2c4ff")
                painter.text((700, y + 5), f"BM25 {hit.score:.4f}", font=mini, fill="#8fe0b4")
                painter.line((95, y + 56, 1000, y + 56), fill="#2f4167", width=1)
            output.append(frame)
        destination = Path("docs/search_demo.gif")
        output[0].save(destination, save_all=True, append_images=output[1:], duration=[1800, 1800, 2400], loop=0, optimize=True)
        print(destination)
    finally:
        model.close()


if __name__ == "__main__":
    main()
