def get_input(prompt_text, required=False, valid_values=None):
    while True:
        value = input(prompt_text).strip()
        if required and not value:
            print("⚠ This field is required. Please enter a value.")
        elif valid_values and value not in valid_values:
            print(f"⚠ Invalid input! Valid options are: {valid_values}")
        else:
            return value if value else None

def parse_bbox(bbox_str):
    try:
        values = [int(x) for x in bbox_str.split(',')]
        if len(values) != 4:
            raise ValueError
        return values
    except:
        print("⚠ Invalid bbox format. Please enter 4 comma-separated integers (e.g., 10,20,100,150).")
        return None

scenes = []

while True:
    print("\n=== Enter information for a new scene ===")

    image_path = get_input("Enter image path (image_path): ", required=True)
    prompt = get_input("Enter description prompt: ", required=True)

    mode_input = get_input("Enter mode (1 = single_file, 2 = multi_file): ", required=True, valid_values=['1', '2'])
    mode = "single_file" if mode_input == '1' else "multi_file"

    scene = {
        'image_path': image_path,
        'prompt': prompt,
        'mode': mode
    }

    if mode == "single_file":
        audio_path_1 = get_input("Enter audio_path_1: ", required=True)
        scene['audio_path_1'] = audio_path_1
        scene['audio_path_2'] = None
        scene['audio_type'] = None
        scene['bbox1'] = None
        scene['bbox2'] = None
    else:  # multi_file
        audio_path_1 = get_input("Enter audio_path_1: ", required=True)
        audio_path_2 = get_input("Enter audio_path_2: ", required=True)
        audio_type = get_input("Enter audio_type (add or para): ", required=True, valid_values=['add', 'para'])

        while True:
            bbox1_str = get_input("Enter bbox1 (format: x1,y1,x2,y2): ", required=True)
            bbox1 = parse_bbox(bbox1_str)
            if bbox1 is not None:
                break

        while True:
            bbox2_str = get_input("Enter bbox2 (format: x1,y1,x2,y2): ", required=True)
            bbox2 = parse_bbox(bbox2_str)
            if bbox2 is not None:
                break

        scene.update({
            'audio_path_1': audio_path_1,
            'audio_path_2': audio_path_2,
            'audio_type': audio_type,
            'bbox1': bbox1,
            'bbox2': bbox2
        })

    scenes.append(scene)

    cont = input("Do you want to add another scene? (y/n): ").strip().lower()
    if cont != 'y':
        break

print("\nList of scenes entered:")
for i, s in enumerate(scenes, 1):
    print(f"Scene {i}: {s}")
