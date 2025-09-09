from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pickle
import dearpygui.dearpygui as dpg
import math
import Parser_v1
from scipy.ndimage import zoom
from scipy.ndimage import rotate
import time
import imageio.v2 as imageio
from PIL import Image

@dataclass
class DicomSliceInfo:
    filepath: str
    transfer_syntax_uid: Optional[str] = None
    slice_thickness: Optional[float] = None
    image_position_patient: Optional[Tuple[float, float, float]] = None
    image_orientation_patient: Optional[List[float]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    pixel_spacing: Optional[Tuple[float, float]] = None
    bits_allocated: Optional[int] = None
    bits_stored: Optional[int] = None
    pixel_representation: Optional[int] = None  # 0 = unsigned, 1 = signed
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_intercept: Optional[float] = None
    rescale_slope: Optional[float] = None
    gantry_tilt: Optional[float] = None
    pixel_data: Optional[np.ndarray] = None


# wczytuje folder z plikami DICOM
def load_folder(slices):
    picked_folder = Parser_v1.open_folder()
    dicom_list = Parser_v1.load_dicom_files_from_folder(picked_folder)
    if not dicom_list:
        dpg.configure_item("pop_up_folder", show=True)
        return
    x = slices.copy()
    slices.clear()
    for path in dicom_list:
        info = Parser_v1.parse_dicom(path)
        if info is None:
            continue
        slices.append(info)
    if not slices:
        slices = x.copy()
        dpg.configure_item("pop_up_folder", show=True)
        return
    save = picked_folder.split("/")[-1]
    print("------------------", save)
    with open(f"{save}.pkl", "wb") as f:
        pickle.dump(slices, f)
        print("Zapisano pkl")
    dpg.set_value("load_folder_text", "Wczytano dane z folderu.")
    dpg.set_value("load_pickle_text", "")
    store_data()


# wczytuje plik pickle
def load_pickle(slices):
    picked_file = Parser_v1.open_file()
    print(picked_file)
    if not picked_file:
        return
    x = slices.copy()
    slices.clear()
    with open(picked_file, "rb") as f:
        y = pickle.load(f)
        for i in y:
            slices.append(i)
    if not isinstance(slices[1], DicomSliceInfo):
        slices = x.copy()
    dpg.set_value("load_pickle_text", "Wczytano dane z pliku.")
    dpg.set_value("load_folder_text", "")
    store_data()


# zapisuje ważne dane
def store_data():
    slices.sort(key=lambda s: s.image_position_patient[2], reverse=True)
    print("============================================")
    z_slice = slices[0]
    global final_columns
    final_columns = z_slice.columns
    global final_rows
    final_rows = z_slice.rows
    global final_depth
    final_depth = len(slices)
    print(final_depth)
    global gantry_tilt
    gantry_tilt = getattr(z_slice, "gantry_tilt", 0.0)
    global pixel_spacing
    pixel_spacing = getattr(z_slice, "pixel_spacing", [1.0, 1.0])
    global pixel_spacing_x
    pixel_spacing_x = pixel_spacing[0]
    global pixel_spacing_y
    pixel_spacing_y = pixel_spacing[1]
    global pixel_spacing_z
    pixel_spacing_z = getattr(z_slice, "slice_thickness", 1.0)
    global zy_fin
    zy_fin = final_columns * pixel_spacing_y
    global zx_fin
    zx_fin = final_rows * pixel_spacing_x
    global xy_fin
    xy_fin = final_rows * pixel_spacing_y
    global xz_fin
    xz_fin = final_depth * pixel_spacing_z
    global yx_fin
    yx_fin = final_columns * pixel_spacing_x
    global yz_fin
    yz_fin = final_depth * pixel_spacing_z
    global current_palette
    current_palette = PALETTES["Grayscale"]
    global wc
    global ww
    global default_wc
    global default_ww
    wc = (
        z_slice.window_center[0]
        if isinstance(z_slice.window_center, list)
        else z_slice.window_center or 40
    )
    ww = (
        z_slice.window_width[0]
        if isinstance(z_slice.window_width, list)
        else z_slice.window_width or 80
    )
    default_wc = wc
    default_ww = ww

    global default_rescale_slope
    default_rescale_slope = z_slice.rescale_slope or 1.0
    global default_rescale_intercept
    default_rescale_intercept = z_slice.rescale_intercept or 0.0
    global max_pixel_value
    max_pixel_value = int(np.max([np.max(s.pixel_data) for s in slices]))
    global polyline_points
    polyline_points = {
        "drawlist_z": [],
        "drawlist_x": [],
        "drawlist_y": [],
    }
    global image_square
    image_square = 512
    print("git")
    first_update()


def update_texture_z(sender, app_data):
    if dpg.get_value("animation_checkbox"):
        return
    if dpg.get_value("projection_checkbox"):
        match dpg.get_value("projection_combo"):
            case "None":
                z_slice = slices[int(final_depth/2)]
                z_projection = z_slice.pixel_data
            case "Average":
                stack = np.stack([sl.pixel_data for sl in slices], axis=0)
                z_projection = np.mean(stack, axis=0)
            case "Maximum":
                stack = np.stack([sl.pixel_data for sl in slices], axis=0)
                z_projection = np.max(stack, axis=0)
            case "First Hit":
                stack = np.stack([s.pixel_data for s in slices], axis=0)
                threshold = dpg.get_value("first_hit_slider")
                fhZ, fhY, fhX = stack.shape
                mask = stack >= threshold
                mask_flipped = mask[::-1, :, :]  # odwraca oś Z
                first_idx_flipped = np.argmax(mask_flipped, axis=0)
                first_idx = fhZ - 1 - first_idx_flipped
                hit_exists = np.any(mask, axis=0)
                z_projection = stack[first_idx, np.arange(fhY)[:, None], np.arange(fhX)]
                z_projection[~hit_exists] = 0
        z_gray = to_grayscale_image(
            z_projection,
            wc,
            ww,
            default_rescale_slope,
            default_rescale_intercept,
        )
        z_rgba = current_palette[z_gray]
        z_texture_data = z_rgba.flatten() / 255.0
        dpg.set_value("dicom_texture_z", z_texture_data)
    else:
        if not sender:
            z_index = 0
        else:
            z_index = dpg.get_value("slider_z")  # to dziala szybciej niż app_data
        z_slice = slices[z_index]
        z_gray = to_grayscale_image(
            z_slice.pixel_data,
            wc,
            ww,
            default_rescale_slope,
            default_rescale_intercept,
        )
        z_rgba = current_palette[z_gray]
        z_texture_data = z_rgba.flatten() / 255.0

        if not sender and not dpg.does_item_exist("dicom_texture_z"):
            dpg.add_dynamic_texture(
                width=final_columns,
                height=final_rows,
                default_value=z_texture_data,
                tag="dicom_texture_z",
                parent="texture_registry",
            )
        elif not sender and dpg.does_item_exist("dicom_texture_z"):
            dpg.delete_item("dicom_texture_z")
            if dpg.does_alias_exist("dicom_texture_z"):
                dpg.remove_alias("dicom_texture_z")
            if dpg.does_item_exist("image_z"):
                dpg.delete_item("image_z")
            if dpg.does_alias_exist("image_z"):
                dpg.remove_alias("image_z")
            if dpg.does_item_exist("marker_layer_z"):
                dpg.delete_item("marker_layer_z")
            if dpg.does_alias_exist("marker_layer_z"):
                dpg.remove_alias("marker_layer_z")
            dpg.add_dynamic_texture(
                width=final_columns,
                height=final_rows,
                default_value=z_texture_data,
                tag="dicom_texture_z",
                parent="texture_registry",
            )
        else:
            dpg.set_value("dicom_texture_z", z_texture_data)

def shear_volume_with_tilt(
    stack: np.ndarray,
    gantry_tilt: float,
    pixel_spacing_z: float,
    pixel_spacing_x: float
) -> np.ndarray:
    """
    Przesuwa dane w wolumenie (Z, Y, X) zgodnie z gantry tilt.
    Górna połowa w lewo, dolna w prawo. Puste miejsca = 0.
    """
    Z, Y, X = stack.shape
    out = np.zeros_like(stack)

    # przesunięcie w pikselach na jednostkę "wysokości"
    offset_per_row = math.sin(math.radians(gantry_tilt)) * (pixel_spacing_z / pixel_spacing_x)
    center_y = Y // 2

    for y in range(Y):
        rel = (y - center_y)  # odległość od środka wiersza
        shift = int(round(rel * offset_per_row))

        if shift < 0:  # w lewo
            out[:, y, :X+shift] = stack[:, y, -shift:]
        elif shift > 0:  # w prawo
            out[:, y, shift:] = stack[:, y, :X-shift]
        else:
            out[:, y, :] = stack[:, y, :]

    return out

def update_texture_x(sender, app_data):
    if dpg.get_value("animation_checkbox"):
        return
    if dpg.get_value("projection_checkbox"):
        match dpg.get_value("projection_combo"):
            case "None":
                x_projection = get_slice_X(
                    slices, int(final_columns/3), gantry_tilt, pixel_spacing_z, pixel_spacing_x
                )
            case "Average":
                stack = np.stack([sl.pixel_data for sl in slices], axis=0)
                print(stack.shape)
                sheared = shear_volume_with_tilt(
                    stack,
                    gantry_tilt=gantry_tilt,
                    pixel_spacing_z=pixel_spacing_z,
                    pixel_spacing_x=pixel_spacing_x
                )
                x_projection = np.mean(sheared, axis=2)
            case "Maximum":
                stack = np.stack([sl.pixel_data for sl in slices], axis=0)
                sheared = shear_volume_with_tilt(
                    stack,
                    gantry_tilt=gantry_tilt,
                    pixel_spacing_z=pixel_spacing_z,
                    pixel_spacing_x=pixel_spacing_x
                )
                
                x_projection = np.max(stack, axis=2)

            case "First Hit":
                stack = np.stack([s.pixel_data for s in slices], axis=0)
                threshold = dpg.get_value("first_hit_slider")
                fhZ, fhY, fhX = stack.shape
                mask = stack >= threshold
                mask_flipped = mask[:, :, ::-1]
                first_idx_flipped = np.argmax(mask_flipped, axis=2)
                first_idx = fhX - 1 - first_idx_flipped
                hit_exists = np.any(mask, axis=2)

                x_projection = stack[np.arange(fhZ)[:, None], np.arange(fhY)[None, :], first_idx]
                x_projection[~hit_exists] = 0
        x_gray = to_grayscale_image(
            x_projection,
            wc,
            ww,
            default_rescale_slope,
            default_rescale_intercept,
        )
        x_rgba = current_palette[x_gray]
        x_texture_data = x_rgba.flatten() / 255.0
        dpg.set_value("dicom_texture_x", x_texture_data)
    else:
        if not sender:
            x_index = 0
        else:
            x_index = dpg.get_value("slider_x")
        x_slice = get_slice_X(
            slices, x_index, gantry_tilt, pixel_spacing_z, pixel_spacing_x
        )
        x_gray = to_grayscale_image(
            x_slice, wc, ww, default_rescale_slope, default_rescale_intercept
        )
        x_rgba = current_palette[x_gray]
        x_texture_data = x_rgba.flatten() / 255.0

        if not sender and not dpg.does_item_exist("dicom_texture_x"):
            dpg.add_dynamic_texture(
                width=final_rows,
                height=final_depth,
                default_value=x_texture_data,
                tag="dicom_texture_x",
                parent="texture_registry",
            )
        elif not sender and dpg.does_item_exist("dicom_texture_x"):
            dpg.delete_item("dicom_texture_x")
            if dpg.does_alias_exist("dicom_texture_x"):
                dpg.remove_alias("dicom_texture_x")
            if dpg.does_item_exist("image_x"):
                dpg.delete_item("image_x")
            if dpg.does_alias_exist("image_x"):
                dpg.remove_alias("image_x")
            if dpg.does_item_exist("marker_layer_x"):
                dpg.delete_item("marker_layer_x")
            if dpg.does_alias_exist("marker_layer_x"):
                dpg.remove_alias("marker_layer_x")
            dpg.add_dynamic_texture(
                width=final_rows,
                height=final_depth,
                default_value=x_texture_data,
                tag="dicom_texture_x",
                parent="texture_registry",
            )
        else:
            dpg.set_value("dicom_texture_x", x_texture_data)


def get_slice_X(slices, x_index, gantry_tilt, pixel_spacing_z, pixel_spacing_x):
    if abs(gantry_tilt) < 1e-3:  # Jeżeli gantry tilt ≈ 0
        return np.array([slice.pixel_data[:, x_index] for slice in slices])
    sin_tilt = math.sin(math.radians(gantry_tilt))
    output = []
    offset = int(round(sin_tilt * pixel_spacing_z / pixel_spacing_x))
    for z_idx, slice in enumerate(slices):
        centered_z = final_depth - z_idx
        x_with_offset = x_index - offset
        x_with_offset = np.clip(x_with_offset, 0, slice.pixel_data.shape[1] - 1)
        x = slice.pixel_data[:, x_with_offset]
        if centered_z * 2 > final_depth:
            output.append(
                np.concatenate(
                    [
                        x[centered_z - z_idx :],
                        np.zeros(abs(z_idx - centered_z), dtype=x.dtype),
                    ]
                )
            )
        elif centered_z * 2 == final_depth:
            output.append(x)
        else:
            output.append(
                np.concatenate(
                    [
                        np.zeros(z_idx - centered_z, dtype=x.dtype),
                        x[: centered_z - z_idx],
                    ]
                )
            )
    return np.array(output)


def update_texture_y(sender, app_data):
    if dpg.get_value("animation_checkbox"):
        return
    if dpg.get_value("projection_checkbox"):
        match dpg.get_value("projection_combo"):
            case "None":
                y_projection = get_slice_Y(
                    slices, int(final_rows/2), gantry_tilt, pixel_spacing_z, pixel_spacing_x
                )
            case "Average":
                stack = np.stack([sl.pixel_data for sl in slices], axis=0)
                y_projection = np.mean(stack, axis=1)
            case "Maximum":
                stack = np.stack([sl.pixel_data for sl in slices], axis=0)
                y_projection = np.max(stack, axis=1)
            case "First Hit":
                stack = np.stack([s.pixel_data for s in slices], axis=0)
                threshold = dpg.get_value("first_hit_slider")
                fhZ, fhY, fhX = stack.shape
                mask = stack >= threshold
    
                first_idx = np.argmax(mask, axis=1)
                hit_exists = np.any(mask, axis=1)

                y_projection = stack[np.arange(fhZ)[:, None], first_idx, np.arange(fhX)]
                y_projection[~hit_exists] = 0
        y_gray = to_grayscale_image(
            y_projection,
            wc,
            ww,
            default_rescale_slope,
            default_rescale_intercept,
        )
        y_rgba = current_palette[y_gray]
        y_texture_data = y_rgba.flatten() / 255.0
        dpg.set_value("dicom_texture_y", y_texture_data)
    else:
        if not sender:
            y_index = 0
        else:
            y_index = dpg.get_value("slider_y")
        y_slice = get_slice_Y(
            slices, y_index, gantry_tilt, pixel_spacing_z, pixel_spacing_x
        )
        y_gray = to_grayscale_image(
            y_slice, wc, ww, default_rescale_slope, default_rescale_intercept
        )
        y_rgba = current_palette[y_gray]
        y_texture_data = y_rgba.flatten() / 255.0

        if not sender and not dpg.does_item_exist("dicom_texture_y"):
            dpg.add_dynamic_texture(
                width=final_rows,
                height=final_depth,
                default_value=y_texture_data,
                tag="dicom_texture_y",
                parent="texture_registry",
            )
        elif not sender and dpg.does_item_exist("dicom_texture_y"):
            dpg.delete_item("dicom_texture_y")
            if dpg.does_alias_exist("dicom_texture_y"):
                dpg.remove_alias("dicom_texture_y")
            if dpg.does_item_exist("image_y"):
                dpg.delete_item("image_y")
            if dpg.does_alias_exist("image_y"):
                dpg.remove_alias("image_y")
            if dpg.does_item_exist("marker_layer_y"):
                dpg.delete_item("marker_layer_y")
            if dpg.does_alias_exist("marker_layer_y"):
                dpg.remove_alias("marker_layer_y")
            dpg.add_dynamic_texture(
                width=final_rows,
                height=final_depth,
                default_value=y_texture_data,
                tag="dicom_texture_y",
                parent="texture_registry",
            )
        else:
            dpg.set_value("dicom_texture_y", y_texture_data)


def get_slice_Y(slices, y_index, gantry_tilt, pixel_spacing_z, pixel_spacing_y):
    if abs(gantry_tilt) < 1e-3:  # Jeżeli gantry tilt ≈ 0
        return np.array([slice.pixel_data[y_index, :] for slice in slices])
    sin_tilt = math.sin(math.radians(gantry_tilt))
    output = []
    for z_idx, slice in enumerate(slices):
        offset = int(round(sin_tilt * z_idx * pixel_spacing_z / pixel_spacing_y))
        y_with_offset = y_index + offset
        y_with_offset = np.clip(y_with_offset, 0, slice.pixel_data.shape[0] - 1)
        output.append(slice.pixel_data[y_with_offset, :])
    return np.array(output)


def change_settings():
    pass


def first_update():
    dpg.configure_item("projection_checkbox", default_value=False)
    dpg.configure_item("animation_checkbox", default_value=False)
    dpg.configure_item("slider_z", max_value=final_depth - 1)
    update_texture_z(0, 0)
    dpg.draw_image(
        texture_tag="dicom_texture_z",
        pmin=(0, 0),
        pmax=(image_square, image_square),
        uv_min=(0, 0),
        uv_max=(1, 1),
        tag="image_z",
        parent="drawlist_z",
    )
    dpg.add_draw_layer(tag="marker_layer_z", parent="drawlist_z")
    dpg.configure_item("slider_x", max_value=final_columns - 1)
    update_texture_x(0, 0)
    dpg.draw_image(
        texture_tag="dicom_texture_x",
        pmin=(0, 0),
        pmax=(image_square, image_square),
        uv_min=(0, 0),
        uv_max=(1, 1),
        tag="image_x",
        parent="drawlist_x",
    )
    dpg.add_draw_layer(tag="marker_layer_x", parent="drawlist_x")
    dpg.configure_item("slider_y", max_value=final_rows - 1)
    update_texture_y(0, 0)
    dpg.draw_image(
        texture_tag="dicom_texture_y",
        pmin=(0, 0),
        pmax=(image_square, image_square),
        uv_min=(0, 0),
        uv_max=(1, 1),
        tag="image_y",
        parent="drawlist_y",
    )
    dpg.add_draw_layer(tag="marker_layer_y", parent="drawlist_y")
    dpg.configure_item("three_view_show", show=True)

    dpg.configure_item(
        "window_center_slider",
        min_value=-max_pixel_value,
        max_value=max_pixel_value,
        default_value=int(default_wc),
        enabled=True,
    )
    dpg.configure_item(
        "window_width_slider",
        max_value=2 * max_pixel_value,
        default_value=int(default_ww),
        enabled=True,
    )
    dpg.configure_item("draw_point_checkbox", enabled=True)
    dpg.configure_item("draw_polyline_checkbox", enabled=True)
    dpg.configure_item("clear_lines_button", enabled=True)
    dpg.configure_item("close_polyline_checkbox", enabled=True)
    dpg.configure_item("projection_checkbox", enabled=True)

    dpg.configure_item("first_hit_slider", min_value=0, max_value=max_pixel_value)


def update_texture(sender, app_data):
    globals()["wc"] = dpg.get_value("window_center_slider")
    globals()["ww"] = dpg.get_value("window_width_slider")
    update_texture_z(1, None)
    update_texture_x(1, None)
    update_texture_y(1, None)


def on_click_any_drawlist(sender, app_data):
    if dpg.get_value("draw_point_checkbox"):
        for axis in ["z", "x", "y"]:
            drawlist_tag = f"drawlist_{axis}"
            if dpg.is_item_hovered(drawlist_tag):
                local_pos = dpg.get_mouse_pos(local=True)  # pozycja względem drawlista
                if axis == "z":
                    z = dpg.get_value("slider_z")
                    x = int(local_pos[0] - 8.0)
                    y = int(local_pos[1] - 56.0)
                    dpg.set_value("slider_z", z)
                    dpg.set_value("slider_x", x)
                    dpg.set_value("slider_y", y)
                    draw_marker("marker_layer_z", (x, y))
                    draw_marker("marker_layer_x", (y, z * (image_square / final_depth)))
                    draw_marker("marker_layer_y", (x, z * (image_square / final_depth)))
                elif axis == "x":
                    z = int(local_pos[1] - 56.0)
                    x = dpg.get_value("slider_x")
                    y = int(local_pos[0] - 528.0)
                    if image_square > final_depth:
                        dpg.set_value("slider_z", z / (image_square / final_depth))
                        dpg.set_value("slider_x", x)
                        dpg.set_value("slider_y", y)
                        draw_marker("marker_layer_z", (x, y))
                        draw_marker("marker_layer_x", (y, z))
                        draw_marker("marker_layer_y", (x, z))
                    else:
                        dpg.set_value("slider_z", z / (image_square / final_depth))
                        dpg.set_value("slider_x", x)
                        dpg.set_value("slider_y", y)
                        draw_marker("marker_layer_z", (x, y))
                        draw_marker("marker_layer_x", (y, z))
                        draw_marker("marker_layer_y", (x, z))
                elif axis == "y":
                    z = int(local_pos[1] - 56.0)
                    x = int(local_pos[0] - 1048.0)
                    y = dpg.get_value("slider_y")
                    if image_square > final_depth:
                        dpg.set_value("slider_z", z / (image_square / final_depth))
                        dpg.set_value("slider_x", x)
                        dpg.set_value("slider_y", y)
                        draw_marker("marker_layer_z", (x, y))
                        draw_marker("marker_layer_x", (y, z))
                        draw_marker("marker_layer_y", (x, z))
                    else:
                        dpg.set_value("slider_z", z / (image_square / final_depth))
                        dpg.set_value("slider_x", x)
                        dpg.set_value("slider_y", y)
                        draw_marker("marker_layer_z", (x, y))
                        draw_marker("marker_layer_x", (y, z))
                        draw_marker("marker_layer_y", (x, z))
                else:
                    continue

                # Zabezpieczenie przed wyjściem poza zakres
                z = max(0, min(image_square - 1, z))
                x = max(0, min(image_square - 1, x))
                y = max(0, min(image_square - 1, y))
                print("-------")

                # Odświeżenie tekstur
                update_texture(None, None)
                break
    if dpg.get_value("draw_polyline_checkbox"):
        for axis in ["z", "x", "y"]:
            drawlist_tag = f"drawlist_{axis}"
            if dpg.is_item_hovered(drawlist_tag):
                if drawlist_tag not in polyline_points:
                    polyline_points[drawlist_tag] = []
                local_pos = dpg.get_mouse_pos(local=True)
                print(local_pos)
                if axis == "z":
                    polyline_points[drawlist_tag].append(
                        [local_pos[0] - 8, local_pos[1] - 56]
                    )
                if axis == "x":
                    polyline_points[drawlist_tag].append(
                        [local_pos[0] - 528, local_pos[1] - 56]
                    )
                if axis == "y":
                    polyline_points[drawlist_tag].append(
                        [local_pos[0] - 1048, local_pos[1] - 56]
                    )
                print(polyline_points[drawlist_tag][-1])
                redraw_polyline(drawlist_tag)


def show_measurements(drawlist, points):
    spacing = (
        pixel_spacing_x,
        pixel_spacing_y,
    )
    if drawlist == "drawlist_z":
        move = (final_columns / image_square, final_rows / image_square)
    elif drawlist == "drawlist_x":
        move = (final_columns / image_square, final_depth / image_square)
    elif drawlist == "drawlist_y":
        move = (final_depth / image_square, final_rows / image_square)
    length = 0.0
    print("spacing:", spacing)
    print("move:", move)
    for i in range(len(points) - 1):
        print("Punkt 1:", (points[i][0] / move[0]), (points[i][1] / move[1]))
        print("Punkt 2:", (points[i + 1][0] / move[0]), (points[i + 1][1] / move[1]))
        dx = ((points[i + 1][0] / move[0]) - (points[i][0] / move[0])) * spacing[0]
        dy = ((points[i + 1][1] / move[1]) - (points[i][1] / move[1])) * spacing[1]
        print(dx, " ", dy)
        length += (dx**2 + dy**2) ** 0.5

    text = f"Dlugosc: {length:.2f} mm"
    if dpg.get_value("close_polyline_checkbox") and len(points) > 2:
        area = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x1 /= move[0]
            y1 /= move[1]
            x2, y2 = points[(i + 1) % len(points)]
            x2 /= move[0]
            y2 /= move[1]
            area += x1 * y2 - x2 * y1
        area = abs(area) * 0.5 * spacing[0] * spacing[1]
        text += f"\nPole: {area:.2f} mm^2"
    dpg.set_value(f"{drawlist}_measurements", text)


def redraw_polyline(drawlist):
    polyline_tag = f"{drawlist}_polyline"
    if dpg.does_item_exist(polyline_tag):
        dpg.delete_item(polyline_tag)
    with dpg.draw_node(parent=drawlist, tag=polyline_tag):
        points = polyline_points[drawlist]
        if len(points) > 1:
            for i in range(len(points) - 1):
                dpg.draw_line(points[i], points[i + 1], color=(0, 255, 0), thickness=2)
                if dpg.get_value("close_polyline_checkbox") and len(points) > 2:
                    dpg.draw_line(points[-1], points[0], color=(0, 255, 0), thickness=2)
    show_measurements(drawlist, points)


def clear_all_polylines():
    for dl in polyline_points:
        polyline_points[dl].clear()
        dpg.delete_item(f"{dl}_polyline", children_only=True)
        dpg.set_value(f"{dl}_measurements", "")


def close_polylines(sender, app_data):
    redraw_polyline("drawlist_z")
    redraw_polyline("drawlist_x")
    redraw_polyline("drawlist_y")


def draw_marker(tag, pos):
    dpg.delete_item(tag, children_only=True)
    dpg.draw_circle(
        center=(pos[0] - 2, pos[1] - 2),
        radius=4,
        color=(255, 0, 0, 255),
        fill=(255, 0, 0, 128),
        parent=tag,
    )


def to_grayscale_image(
    array: np.ndarray, window_center, window_width, slope, intercept
):
    array = array * slope + intercept
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    img = np.clip(array, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255.0
    return img.astype(np.uint8)


def make_grayscale_lut():
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        lut[i] = [i, i, i, 255]
    return lut


def make_hot_lut():
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        lut[i, 0] = min(255, i * 3)
        lut[i, 1] = min(255, max(0, (i - 85) * 3))
        lut[i, 2] = min(255, max(0, (i - 170) * 3))
        lut[i, 3] = 255
    return lut


def make_cool_lut():
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        lut[i, 0] = int(255 * i / 255)
        lut[i, 1] = int(255 * (255 - i) / 255)
        lut[i, 2] = 255
        lut[i, 3] = 255
    return lut


def make_jet_lut():
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        r = int(255 * np.clip(1.5 - abs((i / 255) * 4 - 3), 0, 1))
        g = int(255 * np.clip(1.5 - abs((i / 255) * 4 - 2), 0, 1))
        b = int(255 * np.clip(1.5 - abs((i / 255) * 4 - 1), 0, 1))
        lut[i] = [r, g, b, 255]
    return lut


def make_bone_lut():
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        val = int(i * 0.87 + i * 0.13 * (i / 255))
        lut[i] = [val, val, min(255, int(val * 1.1)), 255]
    return lut


def set_palette_callback(sender, app_data, user_data):
    global current_palette
    selected = app_data
    current_palette = PALETTES[selected]
    update_texture(None, None)

def set_projection_mode(sender, app_data):
    print(app_data)
    update_texture(None,None)

def set_first_hit(sender, app_data):
    # print(app_data)
    update_texture(None,None)
    
def projection_checkbox(sender, app_data):
    if app_data:
        dpg.configure_item("projection_combo", enabled=True)
        dpg.configure_item("slider_z", enabled=False)
        dpg.configure_item("slider_x", enabled=False)
        dpg.configure_item("slider_y", enabled=False)
        dpg.configure_item("first_hit_slider", enabled=True)
        dpg.configure_item("animation_checkbox", enabled=True)
    else:
        dpg.configure_item("projection_combo", enabled=False)
        dpg.configure_item("slider_z", enabled=True)
        dpg.configure_item("slider_x", enabled=True)
        dpg.configure_item("slider_y", enabled=True)
        dpg.configure_item("first_hit_slider", enabled=False)
        dpg.configure_item("animation_checkbox", enabled=False)
    update_texture(None,None)

def animation_checkbox(sender, app_data):
    if app_data:
        dpg.configure_item("animation_window", show=True)
        dpg.configure_item("three_view_show", show=False)
        dpg.configure_item("animation_settings", show=True)
    else:
        dpg.configure_item("animation_window", show=False)
        dpg.configure_item("three_view_show", show=True)
        dpg.configure_item("animation_settings", show=False)

def generate_animation(sender, app_data):
    dpg.set_value("load_animation_text", "Generuje animacje.")
    dpg.configure_item("animation_button", enabled=False)
    t_start = time.perf_counter()
    volume = np.stack([sl.pixel_data for sl in slices], axis=0)  # shape: (Z, Y, X)
    factors = [n/o for n, o in zip( (volume.shape[0], 128, 128), volume.shape)]
    volume_small = zoom(volume, zoom=factors, order=1)
    # num_frames = 36
    num_frames = dpg.get_value("angle_slider")
    angles = np.linspace(0, 360, int(360/num_frames), endpoint=False)
    print("angles",angles)
    global frames
    global animation_speed
    animation_speed = dpg.get_value("fps_slider")
    global direction
    global acceleration
    direction = 1
    acceleration = 1
    frames = []
    for angle in angles:
        rotated_volume = rotate(volume_small, angle=angle, axes=(2, 1), reshape=False, order=1)
        match dpg.get_value("projection_combo"):
            case "None":
                dpg.set_value("load_animation_text", "Blad generowania. Wybierz inny tryb.")
                dpg.configure_item("animation_button", enabled=True)
                return
            case "Average":
                animation_projection = np.mean(rotated_volume, axis=1)
            case "Maximum":
                animation_projection = np.max(rotated_volume, axis=1)
            case "First Hit":
                threshold = dpg.get_value("first_hit_slider")
                fhZ, fhY, fhX = rotated_volume.shape
                mask = rotated_volume >= threshold
    
                first_idx = np.argmax(mask, axis=1)
                hit_exists = np.any(mask, axis=1)

                animation_projection = rotated_volume[np.arange(fhZ)[:, None], first_idx, np.arange(fhX)]
                animation_projection[~hit_exists] = 0
        animation_gray = to_grayscale_image(
            animation_projection,
            wc,
            ww,
            default_rescale_slope,
            default_rescale_intercept,
        )
        animation_rgba = current_palette[animation_gray]
        h, w, _ = animation_rgba.shape
        animation_texture_data = animation_rgba.flatten() / 255.0
        frames.append(animation_texture_data)
    print(sender, app_data)
    dpg.configure_item(
        "animation_angle_slider",
        min_value=0,
        max_value=len(frames)-1,
        default_value=0,
        enabled=True
    )
    if not dpg.does_item_exist("dicom_texture_animation"):
        dpg.add_dynamic_texture(
            width=w,
            height=h,
            default_value=frames[0],
            tag="dicom_texture_animation",
            parent="texture_registry",
        )
        dpg.draw_image(
            texture_tag="dicom_texture_animation",
            pmin=(0, 0),
            pmax=(image_square, image_square),
            uv_min=(0, 0),
            uv_max=(1, 1),
            tag="image_animation",
            parent="drawlist_animation",
        )
    else:
        dpg.set_value("dicom_texture_animation", frames[0])
    t1 = time.perf_counter()
    ttext = f"Zakonczono generowanie w czasie {(t1 - t_start)*1000:.2f} ms"
    dpg.set_value("load_animation_text", ttext)
    dpg.configure_item("play_checkbox", enabled=True)
    dpg.configure_item("ping-pong_checkbox", enabled=True)
    dpg.configure_item("x0.5_checkbox", enabled=True)
    dpg.configure_item("x1_checkbox", enabled=True)
    dpg.configure_item("x2_checkbox", enabled=True)
    dpg.configure_item("animation_button", enabled=True)
    dpg.configure_item("create_gif", enabled=True)
    
    
def change_animation_frame(sender, app_data):
    dpg.set_value("dicom_texture_animation", frames[dpg.get_value("animation_angle_slider")])

def update_animation(sender, app_data):
    global frame_index
    frame_index = dpg.get_value("animation_angle_slider")
    if not dpg.get_value("play_checkbox"):
        dpg.configure_item("animation_angle_slider", enabled=True)
        return
    else:
        dpg.configure_item("animation_angle_slider", enabled=False)
        if dpg.get_value("ping-pong_checkbox"):
            frame_index += globals()["direction"]
            if frame_index >= len(frames) - 1:
                globals()["direction"] = -1
            elif frame_index <= 0:
                globals()["direction"] = 1
        else:
            frame_index = (frame_index + globals()["direction"]) % len(frames)
        dpg.set_value("dicom_texture_animation", frames[frame_index])
        dpg.set_value("animation_angle_slider", frame_index)

        interval = int(animation_speed/acceleration)  # ms
        dpg.set_frame_callback(dpg.get_frame_count() + interval // 16, update_animation)

def animation_speed_settings(sender, app_data):
    print("speed", sender, app_data)
    match sender:
        case "x0.5_checkbox":
            if app_data:
                globals()["acceleration"] = 0.5
                dpg.set_value("x1_checkbox",False)
                dpg.set_value("x2_checkbox",False)
            else:
                globals()["acceleration"] = 1
                dpg.set_value("x1_checkbox",True)
        case "x1_checkbox":
            if app_data:
                globals()["acceleration"] = 1
                dpg.set_value("x0.5_checkbox",False)
                dpg.set_value("x2_checkbox",False)
            else:
                globals()["acceleration"] = 1
                dpg.set_value("x1_checkbox",True)
        case "x2_checkbox":
            if app_data:
                globals()["acceleration"] = 2
                dpg.set_value("x0.5_checkbox",False)
                dpg.set_value("x1_checkbox",False)
            else:
                globals()["acceleration"] = 1
                dpg.set_value("x1_checkbox",True)

def create_gif(sender, app_data):
    if dpg.get_value("ping-pong_checkbox") and len(globals()["frames"]) > 1:
        xframes = globals()["frames"] + globals()["frames"][-2:0:-1]
        images = []
        for frame in xframes:
            print(min(frame), max(frame))
            arr = np.array(frame, dtype=np.float32)

            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8).reshape((final_depth, 128, 4))
            img = Image.fromarray(arr, mode="RGBA")
            img = img.resize((512,512), Image.Resampling.LANCZOS)
            images.append(img)
        images[0].save(
            "animation.gif",
            save_all=True,
            append_images=images[1:],
            duration=animation_speed*acceleration/len(images),
            loop=0
        )
    else:
        images = []
        for frame in frames:
            print(min(frame), max(frame))
            arr = np.array(frame, dtype=np.float32)

            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8).reshape((final_depth, 128, 4))
            img = Image.fromarray(arr, mode="RGBA")
            img = img.resize((512,512), Image.Resampling.LANCZOS)
            images.append(img)
        images[0].save(
            "animation.gif",
            save_all=True,
            append_images=images[1:],
            duration=animation_speed*acceleration/len(images),
            loop=0
        )


    print("GIF zapisany: animation.gif")

def change_draws(sender, app_data):
    if sender == "draw_point_checkbox":
        dpg.configure_item("draw_polyline_checkbox", default_value=False)
    elif sender == "draw_polyline_checkbox":
        dpg.configure_item("draw_point_checkbox", default_value=False)


PALETTES = {
    "Grayscale": make_grayscale_lut(),
    "Hot": make_hot_lut(),
    "Cool": make_cool_lut(),
    "Jet": make_jet_lut(),
    "Bone": make_bone_lut(),
}

if __name__ == "__main__":
    slices: List[DicomSliceInfo] = []
    dpg.create_context()

    # okienko z bledem
    with dpg.window(
        label="Blad",
        modal=True,
        show=False,
        tag="pop_up_folder",
        no_title_bar=True,
        pos=[700, 500],
    ):
        dpg.add_text("Nie wykryto zadnych plikow DICOM w folderze!")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="OK",
                width=75,
                callback=lambda: dpg.configure_item("pop_up_folder", show=False),
            )

    # ustawienia
    with dpg.window(
        label="Ustawienia",
        show=True,
        pos=[0, 0],
        width=800,
        height=150,
        autosize=True,
        no_close=True,
        tag="settings",
    ):
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                dpg.add_button(
                    label="Wczytaj folder z danymi",
                    callback=lambda: load_folder(slices),
                )
                dpg.add_text("", tag="load_folder_text")
            with dpg.table_row():
                dpg.add_button(
                    label="Wczytaj wczesniej wybrane dane",
                    callback=lambda: load_pickle(slices),
                )
                dpg.add_text("", tag="load_pickle_text")

        dpg.add_slider_int(
            label="Window Center",
            min_value=0,
            max_value=0,
            default_value=0,
            callback=update_texture,
            width=700,
            enabled=False,
            tag="window_center_slider",
        )
        dpg.add_slider_int(
            label="Window Width",
            min_value=1,
            max_value=2,
            default_value=0,
            callback=update_texture,
            width=700,
            enabled=False,
            tag="window_width_slider",
        )
        dpg.add_combo(
            list(PALETTES.keys()),
            label="Wybierz palete kolorow",
            default_value="Grayscale",
            callback=set_palette_callback,
        )
        dpg.add_checkbox(
            label="Rysuj punkt",
            tag="draw_point_checkbox",
            default_value=False,
            enabled=False,
            callback=change_draws,
        )
        dpg.add_checkbox(
            label="Rysuj linie",
            tag="draw_polyline_checkbox",
            default_value=False,
            enabled=False,
            callback=change_draws,
        )
        dpg.add_checkbox(
            label="Zamknij ksztalt (Pole)",
            tag="close_polyline_checkbox",
            default_value=False,
            enabled=False,
            callback=close_polylines,
        )
        dpg.add_button(
            label="Wyczysc linie",
            tag="clear_lines_button",
            enabled=False,
            callback=lambda: clear_all_polylines(),
        )
        dpg.add_separator()
        dpg.add_checkbox(
            label="Projekcja",
            tag="projection_checkbox",
            default_value=False,
            enabled=False,
            callback=projection_checkbox,
        )
        dpg.add_combo(
            items=["None", "Average", "Maximum", "First Hit"],
            label="Wybierz tryb projekcji",
            tag="projection_combo",
            default_value="None",
            enabled=False,
            callback=set_projection_mode,
        )
        dpg.add_slider_int(
            label="Próg First Hit",
            min_value=0,
            max_value=1,
            default_value=0,
            width=700,
            enabled=False,
            tag="first_hit_slider",
            callback=set_first_hit,
        )
        dpg.add_checkbox(
            label="Animacja 3D",
            tag="animation_checkbox",
            default_value=False,
            enabled=False,
            callback=animation_checkbox,
        )

    with dpg.texture_registry(show=False, tag="texture_registry"):
        pass

    with dpg.window(
        show=False,
        label="Widok w trzech przekrojach",
        tag="three_view_show",
        pos=[0, 330],
        width=1800,
        height=1150,
        autosize=True,
        no_close=True,
    ):
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                dpg.add_text("Przekroj (Z)")
                dpg.add_text("Przekroj (X)")
                dpg.add_text("Przekroj (Y)")
            with dpg.table_row():
                dpg.add_slider_int(
                    min_value=0,
                    max_value=1,
                    default_value=0,
                    width=512,
                    tag="slider_z",
                    callback=update_texture_z,
                )
                dpg.add_slider_int(
                    min_value=0,
                    max_value=1,
                    default_value=0,
                    width=512,
                    tag="slider_x",
                    callback=update_texture_x,
                )
                dpg.add_slider_int(
                    min_value=0,
                    max_value=1,
                    default_value=0,
                    width=512,
                    tag="slider_y",
                    callback=update_texture_y,
                )
            with dpg.table_row():
                with dpg.drawlist(
                    width=512,
                    height=512,
                    pos=[0, 0],
                    tag="drawlist_z",
                ):
                    pass
                with dpg.drawlist(
                    width=512,
                    height=512,
                    pos=[0, 0],
                    tag="drawlist_x",
                ):
                    pass
                with dpg.drawlist(
                    width=512,
                    height=512,
                    pos=[0, 0],
                    tag="drawlist_y",
                ):
                    pass
            with dpg.table_row():
                dpg.add_text("Dlugosc: ", tag="drawlist_z_measurements")
                dpg.add_text("Dlugosc: ", tag="drawlist_x_measurements")
                dpg.add_text("Dlugosc: ", tag="drawlist_y_measurements")

    # ustawienia animacji
    with dpg.window(
        label="Ustawienia animacji",
        show=False,
        pos=[830, 0],
        width=800,
        height=150,
        autosize=True,
        no_close=True,
        tag="animation_settings",
    ):
        dpg.add_slider_int(
            label="Kat obrotu",
            min_value=1,
            max_value=360,
            default_value=10,
            width=360,
            enabled=True,
            tag="angle_slider",
        )
        dpg.add_slider_int(
            label="Predkosc",
            min_value=0,
            max_value=3000,
            default_value=1000,
            width=360,
            enabled=True,
            tag="fps_slider",
        )
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                dpg.add_button(
                    label="Stworz animacje",
                    tag="animation_button",
                    enabled=True,
                    callback=generate_animation,
                )
                dpg.add_text("", tag="load_animation_text")


    # okno animacji
    with dpg.window(
        show=False,
        label="Animacja",
        tag="animation_window",
        pos=[0, 330],
        width=800,
        height=1150,
        autosize=True,
        no_close=True,
    ):
        with dpg.drawlist(
            width=512,
            height=512,
            pos=[0, 0],
            tag="drawlist_animation",
        ):
            pass

        dpg.add_slider_int(
            min_value=0,
            max_value=1,
            default_value=0,
            width=512,
            enabled=False,
            tag="animation_angle_slider",
            callback=change_animation_frame
        )
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                dpg.add_checkbox(
                    label="Graj",
                    tag="play_checkbox",
                    enabled=False,
                    default_value=False,
                    callback=update_animation,
                )
                dpg.add_checkbox(
                    label="Ping-pong",
                    tag="ping-pong_checkbox",
                    enabled=False,
                    default_value=False,
                )
                dpg.add_checkbox(
                    label="x0.5",
                    tag="x0.5_checkbox",
                    enabled=False,
                    default_value=False,
                    callback=animation_speed_settings
                )
                dpg.add_checkbox(
                    label="x1",
                    tag="x1_checkbox",
                    enabled=False,
                    default_value=True,
                    callback=animation_speed_settings
                )
                dpg.add_checkbox(
                    label="x2",
                    tag="x2_checkbox",
                    enabled=False,
                    default_value=False,
                    callback=animation_speed_settings
                )
        dpg.add_button(
            label="Stwórz gifa",
            tag="create_gif",
            enabled=False,
            callback=create_gif,
        )


    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=on_click_any_drawlist)

    dpg.create_viewport(
        title="DICOM Slice Viewer",
        width=1920,
        height=1080,
        x_pos=0,
        y_pos=0,
        clear_color=(47, 47, 47, 255),
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
