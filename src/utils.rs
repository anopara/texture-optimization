use std::path::Path;

pub fn save_image_to_file(path: &Path, image: &image::RgbaImage) {
    if let Some(parent_path) = path.parent() {
        let _ = std::fs::create_dir_all(&parent_path);
    }

    match image.save(&path) {
        Err(error) => panic!("There was a problem saving the image: {:?}", error),
        Ok(_) => println!("Saved image at {:?}", path),
    }
}

pub fn load_example_image(path: &str, size: (u32, u32)) -> image::DynamicImage {
    let img = image::open(path).unwrap();
    img.resize(size.0, size.1, image::imageops::Gaussian)
}
