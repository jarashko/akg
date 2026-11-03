import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CORRECT_LOWER = (0, 120, 0)
CORRECT_UPPER = (80, 255, 80)


def parse_rgb(text):
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Ожидается цвет в формате R,G,B")
    try:
        values = tuple(int(item) for item in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Компоненты цвета должны быть целыми числами") from error
    if any(value < 0 or value > 255 for value in values):
        raise argparse.ArgumentTypeError("Компоненты цвета должны быть в диапазоне 0..255")
    return values


def ordered_interval(lower, upper):
    low = tuple(min(left, right) for left, right in zip(lower, upper))
    high = tuple(max(left, right) for left, right in zip(lower, upper))
    return low, high


def load_font(size):
    for candidate in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_pair(foreground_path, background_path):
    foreground = Image.open(foreground_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")
    if background.size != foreground.size:
        print(
            f"Второе изображение {background.size[0]}x{background.size[1]} "
            f"масштабируется до {foreground.size[0]}x{foreground.size[1]}"
        )
        background = background.resize(foreground.size, Image.Resampling.BILINEAR)
    return foreground, background


def chroma_key_native(foreground, background, lower, upper):
    red_low, green_low, blue_low = lower
    red_high, green_high, blue_high = upper
    foreground_pixels = list(foreground.getdata())
    background_pixels = list(background.getdata())
    result = []
    for (red, green, blue), background_pixel in zip(foreground_pixels, background_pixels):
        if (
            red_low <= red <= red_high
            and green_low <= green <= green_high
            and blue_low <= blue <= blue_high
        ):
            result.append(background_pixel)
        else:
            result.append((red, green, blue))
    output = Image.new("RGB", foreground.size)
    output.putdata(result)
    return output


def chroma_key_opencv(foreground, background, lower, upper):
    foreground_array = np.asarray(foreground)
    background_array = np.asarray(background)
    mask = cv2.inRange(
        foreground_array,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8),
    )
    kept = cv2.bitwise_and(foreground_array, foreground_array, mask=cv2.bitwise_not(mask))
    replaced = cv2.bitwise_and(background_array, background_array, mask=mask)
    return cv2.add(kept, replaced)


def measure(func, args, repeats):
    func(*args)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = func(*args)
        samples.append(time.perf_counter() - start)
    samples.sort()
    median = samples[len(samples) // 2]
    return value, median, samples


def format_ms(seconds):
    return f"{seconds * 1000:.2f}"


def draw_person(draw, width, height):
    center_x = width // 2
    center_y = height // 2
    skin = (235, 190, 150)
    shirt = (25, 45, 210)
    legs = (50, 45, 55)
    head_radius = max(8, int(min(width, height) * 0.08))
    head_top = center_y - int(height * 0.34)
    draw.ellipse(
        [center_x - head_radius, head_top, center_x + head_radius, head_top + 2 * head_radius],
        fill=skin,
    )
    body_top = head_top + 2 * head_radius
    body_width = int(width * 0.11)
    body_height = int(height * 0.26)
    draw.rectangle(
        [center_x - body_width, body_top, center_x + body_width, body_top + body_height],
        fill=shirt,
    )
    arm_width = int(width * 0.07)
    arm_height = int(height * 0.16)
    arm_top = body_top + int(height * 0.03)
    draw.rectangle(
        [center_x - body_width - arm_width, arm_top, center_x - body_width, arm_top + arm_height],
        fill=skin,
    )
    draw.rectangle(
        [center_x + body_width, arm_top, center_x + body_width + arm_width, arm_top + arm_height],
        fill=skin,
    )
    leg_width = max(4, int(width * 0.045))
    leg_height = int(height * 0.2)
    gap = max(2, int(width * 0.012))
    leg_top = body_top + body_height
    draw.rectangle(
        [center_x - gap - leg_width, leg_top, center_x - gap, leg_top + leg_height],
        fill=legs,
    )
    draw.rectangle(
        [center_x + gap, leg_top, center_x + gap + leg_width, leg_top + leg_height],
        fill=legs,
    )


def make_foreground(width, height):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grid_x = np.broadcast_to(x, (height, width))
    wave = 6.0 * np.sin(y[:, None] * 30.0)
    red = np.clip(8.0 + 48.0 * grid_x, 0, 255).astype(np.uint8)
    green = np.clip(135.0 + 110.0 * grid_x + wave, 0, 255).astype(np.uint8)
    blue = np.clip(12.0 + 36.0 * grid_x, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.dstack([red, green, blue]))
    draw_person(ImageDraw.Draw(image), width, height)
    return image


def make_background(width, height):
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    sky = y < 0.72
    red = np.where(sky, 80.0 + 90.0 * y + 40.0 * x, 46.0 + 70.0 * (1.0 - y))
    green = np.where(sky, 150.0 + 50.0 * (1.0 - y), 120.0 + 40.0 * (1.0 - y))
    blue = np.where(sky, 235.0 - 70.0 * y, 55.0 + 15.0 * (1.0 - y))
    image = Image.fromarray(np.dstack([red, green, blue]).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    sun = int(min(width, height) * 0.08)
    sun_x = int(width * 0.78)
    sun_y = int(height * 0.15)
    draw.ellipse([sun_x - sun, sun_y - sun, sun_x + sun, sun_y + sun], fill=(255, 214, 70))
    hill_y = int(height * 0.72)
    draw.polygon(
        [
            (0, hill_y + int(height * 0.08)),
            (int(width * 0.28), hill_y - int(height * 0.06)),
            (int(width * 0.55), hill_y + int(height * 0.05)),
            (width, hill_y - int(height * 0.02)),
            (width, height),
            (0, height),
        ],
        fill=(74, 130, 62),
    )
    return image


def mask_image(foreground, lower, upper):
    mask = cv2.inRange(
        np.asarray(foreground),
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8),
    )
    return Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))


def captioned_panel(image, caption, panel_size):
    font = load_font(22)
    canvas = Image.new("RGB", (panel_size[0], panel_size[1] + 42), (255, 255, 255))
    fitted = image.resize(panel_size, Image.Resampling.BILINEAR)
    canvas.paste(fitted, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), caption, fill=(25, 25, 25), font=font)
    return canvas


def save_gallery(panels, path, columns=3):
    gap = 16
    rows = math.ceil(len(panels) / columns)
    cell_w = panels[0].width
    cell_h = panels[0].height
    sheet = Image.new(
        "RGB",
        (columns * cell_w + gap * (columns + 1), rows * cell_h + gap * (rows + 1)),
        (245, 245, 247),
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet.paste(panel, (gap + column * (cell_w + gap), gap + row * (cell_h + gap)))
    sheet.save(path)


def save_benchmark_chart(rows, path):
    width, height = 1000, 580
    image = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    font = load_font(18)
    small = load_font(15)
    draw.text((48, 22), "Время выполнения хромакея", fill=(20, 20, 20), font=title_font)
    left, right, top, bottom = 110, 940, 100, 470
    positive = [value for _, native, library in rows for value in (native, library) if value > 0]
    min_value = min(positive) * 0.65
    max_value = max(positive) * 1.8
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)

    def y_of(value):
        return bottom - (math.log10(max(value, min_value)) - log_min) / (log_max - log_min) * (bottom - top)

    exponent = math.floor(log_min)
    while exponent <= math.ceil(log_max):
        value = 10 ** exponent
        if min_value <= value <= max_value:
            y = y_of(value)
            draw.line([(left, y), (right, y)], fill=(226, 226, 230))
            milliseconds = value * 1000.0
            if milliseconds >= 100:
                label = f"{milliseconds:.0f} мс"
            elif milliseconds >= 10:
                label = f"{milliseconds:.1f} мс"
            else:
                label = f"{milliseconds:.2f} мс"
            draw.text((16, y - 9), label, fill=(90, 90, 90), font=small)
        exponent += 1

    draw.line([(left, top), (left, bottom), (right, bottom)], fill=(35, 35, 35), width=2)
    group_width = (right - left) / len(rows)
    bar_width = min(54, group_width * 0.28)
    native_color = (58, 110, 184)
    library_color = (32, 150, 112)
    for index, (label, native, library) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        for shift, value, color in (
            (-bar_width * 0.7, native, native_color),
            (bar_width * 0.7, library, library_color),
        ):
            x0 = center + shift - bar_width / 2
            y0 = y_of(value)
            draw.rectangle([x0, y0, x0 + bar_width, bottom], fill=color)
            draw.text((x0 - 6, y0 - 22), format_ms(value), fill=(30, 30, 30), font=small)
        text_width = draw.textlength(label, font=font)
        draw.text((center - text_width / 2, bottom + 14), label, fill=(30, 30, 30), font=font)

    draw.rectangle([left, 528, left + 18, 546], fill=native_color)
    draw.text((left + 26, 526), "Чистый Python", fill=(30, 30, 30), font=font)
    draw.rectangle([left + 230, 528, left + 248, 546], fill=library_color)
    draw.text((left + 256, 526), "OpenCV", fill=(30, 30, 30), font=font)
    draw.text((left + 430, 526), "Шкала времени логарифмическая", fill=(90, 90, 90), font=small)
    image.save(path)


def print_timings(rows):
    print(f"{'Разрешение':<14}{'Python, мс':>14}{'OpenCV, мс':>14}{'Ускорение':>12}")
    for label, native, library in rows:
        speedup = native / library if library else float("inf")
        print(f"{label:<14}{format_ms(native):>14}{format_ms(library):>14}{speedup:>11.1f}x")


def ensure_same(native_image, library_array):
    native_array = np.asarray(native_image)
    if native_array.shape != library_array.shape or not np.array_equal(native_array, library_array):
        mismatch = int(np.count_nonzero(np.any(native_array != library_array, axis=2)))
        raise SystemExit(f"Реализации дали разные результаты, различающихся пикселей: {mismatch}")


def run_case(foreground, background, lower, upper, native_repeats, library_repeats):
    native_image, native_time, _ = measure(
        chroma_key_native,
        (foreground, background, lower, upper),
        native_repeats,
    )
    library_array, library_time, _ = measure(
        chroma_key_opencv,
        (foreground, background, lower, upper),
        library_repeats,
    )
    ensure_same(native_image, library_array)
    return native_image, library_array, native_time, library_time


def build_parser():
    parser = argparse.ArgumentParser(
        description="Хромакей: замена областей заданного цветового интервала вторым изображением."
    )
    parser.add_argument("--fg", help="Изображение переднего плана")
    parser.add_argument("--bg", help="Изображение, которым заменяются области ключа")
    parser.add_argument("--lower", type=parse_rgb, help="Нижняя граница интервала R,G,B")
    parser.add_argument("--upper", type=parse_rgb, help="Верхняя граница интервала R,G,B")
    parser.add_argument("--output", default="result.png", help="Куда сохранить результат")
    parser.add_argument("--mask", help="Куда сохранить маску заменённых пикселей")
    parser.add_argument("--benchmark", action="store_true", help="Сравнить время двух реализаций")
    parser.add_argument("--demo", action="store_true", help="Построить примеры и измерить быстродействие")
    parser.add_argument("--native-repeats", type=int, default=3)
    parser.add_argument("--opencv-repeats", type=int, default=30)
    return parser


def ask_rgb(prompt):
    while True:
        try:
            return parse_rgb(input(prompt).strip())
        except argparse.ArgumentTypeError as error:
            print(error)


def interactive_arguments(parser):
    print("Интерактивный режим. Пример зелёного экрана: 0,120,0 и 80,255,80")
    foreground = input("Путь к переднему плану: ").strip().strip('"')
    background = input("Путь ко второму изображению: ").strip().strip('"')
    lower = ask_rgb("Нижняя граница R,G,B: ")
    upper = ask_rgb("Верхняя граница R,G,B: ")
    output = input("Файл результата [result.png]: ").strip().strip('"') or "result.png"
    benchmark = input("Сравнить быстродействие? [y/N]: ").strip().lower() in {"y", "yes", "д", "да"}
    return parser.parse_args(
        [
            "--fg",
            foreground,
            "--bg",
            background,
            "--lower",
            ",".join(str(value) for value in lower),
            "--upper",
            ",".join(str(value) for value in upper),
            "--output",
            output,
        ]
        + (["--benchmark"] if benchmark else [])
    )


def process_files(args):
    lower, upper = ordered_interval(args.lower, args.upper)
    foreground, background = load_pair(args.fg, args.bg)
    print(f"Интервал RGB: {lower} .. {upper}")
    print(f"Размер: {foreground.size[0]}x{foreground.size[1]}")
    if args.benchmark:
        native_image, _, native_time, library_time = run_case(
            foreground,
            background,
            lower,
            upper,
            args.native_repeats,
            args.opencv_repeats,
        )
        print_timings([(f"{foreground.size[0]}x{foreground.size[1]}", native_time, library_time)])
        native_image.save(args.output)
    else:
        Image.fromarray(chroma_key_opencv(foreground, background, lower, upper)).save(args.output)
    if args.mask:
        mask_image(foreground, lower, upper).save(args.mask)
    print(f"Результат сохранён: {args.output}")


def run_demo():
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    width, height = 960, 540
    foreground = make_foreground(width, height)
    background = make_background(width, height)
    foreground.save(output_dir / "foreground.png")
    background.save(output_dir / "background.png")

    cases = [
        ("Узкий интервал", (0, 200, 0), (80, 255, 80), "result_narrow.png"),
        ("Рабочий интервал", CORRECT_LOWER, CORRECT_UPPER, "result.png"),
        ("Широкий интервал", (0, 0, 0), (255, 255, 180), "result_wide.png"),
    ]
    panel_size = (420, 236)
    results = []
    mask = None
    for title, lower, upper, name in cases:
        library = chroma_key_opencv(foreground, background, lower, upper)
        native = chroma_key_native(foreground, background, lower, upper)
        ensure_same(native, library)
        result = Image.fromarray(library)
        result.save(output_dir / name)
        results.append((title, result))
        if name == "result.png":
            mask = mask_image(foreground, lower, upper)
            mask.save(output_dir / "mask.png")
    panels = [
        captioned_panel(foreground, "Передний план", panel_size),
        captioned_panel(background, "Второе изображение", panel_size),
        captioned_panel(mask, "Маска ключа", panel_size),
    ]
    panels.extend(captioned_panel(result, title, panel_size) for title, result in results)
    save_gallery(panels, output_dir / "comparison.png")

    rows = []
    sizes = ((320, 240, 5, 40), (640, 480, 3, 30), (1280, 720, 2, 20))
    print("Проверка совпадения реализаций пройдена.")
    print(f"{'Разрешение':<14}{'Python, мс':>14}{'OpenCV, мс':>14}{'Ускорение':>12}")
    for size_w, size_h, native_repeats, library_repeats in sizes:
        fg = make_foreground(size_w, size_h)
        bg = make_background(size_w, size_h)
        _, _, native_time, library_time = run_case(
            fg,
            bg,
            CORRECT_LOWER,
            CORRECT_UPPER,
            native_repeats,
            library_repeats,
        )
        label = f"{size_w}x{size_h}"
        rows.append((label, native_time, library_time))
        speedup = native_time / library_time
        print(f"{label:<14}{format_ms(native_time):>14}{format_ms(library_time):>14}{speedup:>11.1f}x")
    save_benchmark_chart(rows, output_dir / "benchmark.png")
    lines = ["resolution_native_s_opencv_s_speedup"]
    for label, native_time, library_time in rows:
        lines.append(f"{label} {native_time:.6f} {library_time:.6f} {native_time / library_time:.2f}")
    (output_dir / "timings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Примеры сохранены в {output_dir}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    if len(sys.argv) == 1:
        args = interactive_arguments(parser)
    else:
        args = parser.parse_args()
    if args.demo:
        run_demo()
        return
    if not args.fg or not args.bg or args.lower is None or args.upper is None:
        parser.error("Нужны --fg, --bg, --lower и --upper либо запуск без аргументов или с --demo")
    for path in (args.fg, args.bg):
        if not Path(path).is_file():
            raise SystemExit(f"Файл не найден: {path}")
    process_files(args)


if __name__ == "__main__":
    main()
