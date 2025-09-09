import os
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog

TAGS = {
    (0x0002, 0x0010): "TransferSyntaxUID",  # Tag(2,16)
    (0x0018, 0x0050): "SliceThickness",  # Tag(24,80)
    (0x0018, 0x1120): "Gantry/DetectorTilt",  # Tag(24,4384)
    (0x0020, 0x0032): "ImagePositionPatient",  # Tag(32,50)
    (0x0020, 0x0037): "ImageOrientationPatient",  # Tag(32,55)
    (0x0028, 0x0010): "Rows",  # Tag(40,16)
    (0x0028, 0x0011): "Columns",  # Tag(40,17)
    (0x0028, 0x0030): "PixelSpacing",  # Tag(40,48)
    (0x0028, 0x0100): "BitsAllocated",  # Tag(40,256)
    (0x0028, 0x0101): "BitsStored",  # Tag(40,257)
    (0x0028, 0x0103): "PixelRepresentation",  # Tag(40,259)
    (0x0028, 0x1050): "WindowCenter",  # Tag(40,4176)
    (0x0028, 0x1051): "WindowWidth",  # Tag(40,4177)
    (0x0028, 0x1052): "RescaleIntercept",  # Tag(40,4178)
    (0x0028, 0x1053): "RescaleSlope",  # Tag(40,4179)
}

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

def open_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        print("Wybrano folder:", folder_path)
        return folder_path
    
def open_file():
    file_path = filedialog.askopenfilename(
        title="Wybierz plik",
        filetypes=[("Pliki pickle", "*.pkl")]
    )
    if file_path:
        print("Wybrano plik:", file_path)
        return file_path


def load_dicom_files_from_folder(folder_path):
    dicom_files = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            dicom_files.append(filepath)
    return dicom_files

def parse_dicom(filename):
    with open(filename, "rb") as f:
        f.seek(128)  # Pomijamy preambułę
        if f.read(4) != b"DICM":
            print("Nie znaleziono nagłówka DICM")
            return
        tags = {}
        endian = ""
        while True:
            try:
                tag = read_tag(f)
                vr, length = read_vr(f)
                value = read_value(f, vr, length)
                if tag == (0x0002, 0x0010):
                    endian = get_endianess(value.decode().strip("\x00") or "")
                    # print(endian)
                if tag in TAGS:
                    tags[tag] = value
                if tag == (0x7FE0, 0x0010):
                    num_pixels = int(tags.get((0x0028, 0x0010))) * int(
                        tags.get((0x0028, 0x0011))
                    )
                    raw = value
                    # print(value)
                    if int(tags.get((0x0028, 0x0100))) == 8:
                        fmt = "B" if int(tags.get((0x0028, 0x0103))) == 0 else "b"
                    elif int(tags.get((0x0028, 0x0100))) == 16:
                        fmt = "H" if int(tags.get((0x0028, 0x0103))) == 0 else "h"
                    else:
                        raise NotImplementedError(
                            "BitsAllocated != 8 or 16 nieobsługiwane"
                        )
                    pixel_format = f"{endian}{num_pixels}{fmt}"
                    # print(pixel_format)
                    pixels = struct.unpack(pixel_format, raw)
                    # print(pixels)
                    break
            except Exception:
                break

        return DicomSliceInfo(
            filepath=filename,
            transfer_syntax_uid=tags.get((0x0002, 0x0010), None).decode().strip("\x00"),
            slice_thickness=float(tags.get((0x0018, 0x0050), "nan")),
            image_position_patient=tuple(
                float(v) for v in tags.get((0x0020, 0x0032), "0\\0\\0").split("\\")
            ),
            image_orientation_patient=[
                float(v)
                for v in tags.get((0x0020, 0x0037), "1\\0\\0\\0\\1\\0").split("\\")
            ],
            rows=int(tags.get((0x0028, 0x0010), 0)),
            columns=int(tags.get((0x0028, 0x0011), 0)),
            pixel_spacing=tuple(
                float(v) for v in tags.get((0x0028, 0x0030), "1\\1").split("\\")
            ),
            bits_allocated=int(tags.get((0x0028, 0x0100), 0)),
            bits_stored=int(tags.get((0x0028, 0x0101), 0)),
            pixel_representation=int(tags.get((0x0028, 0x0103), 0)),
            window_center=(
                [float(v) for v in tags.get((0x0028, 0x1050), "0").split("\\")]
                if isinstance(tags.get((0x0028, 0x1050), "0"), str)
                else [float(tags.get((0x0028, 0x1050), "0"))]
            ),
            window_width=(
                [float(v) for v in tags.get((0x0028, 0x1051), "1").split("\\")]
                if isinstance(tags.get((0x0028, 0x1051), "1"), str)
                else [float(tags.get((0x0028, 0x1051), "1"))]
            ),
            rescale_intercept=float(tags.get((0x0028, 0x1052), "0")),
            rescale_slope=float(tags.get((0x0028, 0x1053), "1")),
            gantry_tilt=(
                float(tags.get((0x0018, 0x1120), "0"))
                if (0x0018, 0x1120) in tags
                else None
            ),
            pixel_data=np.array(pixels, dtype=np.int16).reshape(
                (int(tags.get((0x0028, 0x0010), 0)), int(tags.get((0x0028, 0x0011), 0)))
            ),
        )
    
def read_tag(file):
    group, element = struct.unpack("<HH", file.read(4))
    return (group, element)

def read_vr(file):
    vr = file.read(2).decode()
    if vr in ["OB", "OW", "OF", "SQ", "UT", "UN"]:
        file.read(2)
        length = struct.unpack("<I", file.read(4))[0]
    else:
        length = struct.unpack("<H", file.read(2))[0]
    return vr, length

def read_value(file, vr, length):
    if vr == "DS" or vr == "IS":
        return file.read(length).decode().strip()
    elif vr in ["US", "SS"]:
        if length == 2:
            return struct.unpack("<H" if vr == "US" else "<h", file.read(2))[0]
        else:
            return file.read(length)
    elif vr in ["FL"]:
        return struct.unpack("<f", file.read(4))[0]
    elif vr in ["FD"]:
        return struct.unpack("<d", file.read(8))[0]
    else:
        return file.read(length)
    
def get_endianess(transfer_syntax_uid: str) -> str:
    if transfer_syntax_uid == "1.2.840.10008.1.2.2":
        return ">"  # Big endian
    else:
        return "<"  # Little endian (domyślnie)