use image::GenericImage;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use rustdct::DCTplanner;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;

#[derive(Clone, Debug)]
pub struct Coord {
    pub x: u32,
    pub y: u32,
}

impl Coord {
    pub fn new(x: u32, y: u32) -> Self {
        Coord { x, y }
    }

    pub fn distanceSquared(c1: &Coord, c2: &Coord) -> f64 {
        let x1 = c1.x as f64;
        let x2 = c2.x as f64;
        let y1 = c1.y as f64;
        let y2 = c2.y as f64;
        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    }

    pub fn distanceGaussian(c1: &Coord, c2: &Coord, width: f64) -> f64 {
        let x1 = (c1.x as f64 - width / 2.0) / width;
        let x2 = (c2.x as f64 - width / 2.0) / width;
        let y1 = (c1.y as f64 - width / 2.0) / width;
        let y2 = (c2.y as f64 - width / 2.0) / width;
        (0.5 * (x2 - x1).powf(2.0) + 0.5 * (y2 - y1).powf(2.0)).exp()
    }

    pub fn to_flat(&self, dims: (u32, u32)) -> u32 {
        dims.0 * self.y + self.x
    }
}

#[derive(Clone, Debug)]
pub struct TextureNeighbourhood {
    pub center: Coord,
    //original coordinates of this neighbourhood
    pub coords: Vec<Coord>,
    //image data
    pub data: Vec<image::Rgba<u8>>,
}

impl TextureNeighbourhood {
    pub fn new(
        center: Coord,
        coords: Vec<Coord>,
        data: Vec<image::Rgba<u8>>,
    ) -> TextureNeighbourhood {
        TextureNeighbourhood {
            center,
            coords,
            data,
        }
    }

    pub fn into_raw(&self) -> Vec<u8> {
        let mut raw = Vec::new();
        for d in self.data.iter() {
            raw.extend_from_slice(&[d[0], d[1], d[2], d[3]]);
        }
        raw
    }

    pub fn to_float64(&self) -> Vec<f64> {
        self.into_raw()
            .iter()
            .map(|a| (*a as f64) / 255.0)
            .collect()
    }
}

pub fn break_image_into_neighbourhoods(
    image: &image::RgbaImage,
    width: usize,
    offset: u32,
) -> Vec<TextureNeighbourhood> {
    let mut neighbourhoods_vec = Vec::new();
    let image = image.clone();

    let (dimx, dimy) = image.dimensions();

    for i in 0..(dimx / offset + 1) {
        for j in 0..(dimy / offset + 1) {
            let (x, y) = match offset {
                0 | 1 => (i, j),
                _ => (offset * i + offset / 2, offset * j + offset / 2),
            };
            //check that the coordinates will stay valid
            let (x_out, y_out) =
                ensure_neighbourhood_center_validity(x, y, width as u32, (dimx, dimy));
            neighbourhoods_vec.push(get_image_neighbourhood(&image, x_out, y_out, width as u32));
        }
    }
    neighbourhoods_vec
}

fn ensure_neighbourhood_center_validity(
    x: u32,
    y: u32,
    width: u32,
    size: (u32, u32),
) -> (u32, u32) {
    let mut x_out = x;
    let mut y_out = y;
    if (x + width / 2) > size.0 {
        x_out = size.0 - width / 2 - 1;
    }
    if (y + width / 2) > size.1 {
        y_out = size.1 - width / 2 - 1;
    }
    if (x as i32 - width as i32 / 2) < 0 {
        x_out = width / 2;
    }
    if (y as i32 - width as i32 / 2) < 0 {
        y_out = width / 2;
    }
    (x_out, y_out)
}

fn get_image_neighbourhood(
    image: &image::RgbaImage,
    x: u32,
    y: u32,
    width: u32,
) -> TextureNeighbourhood {
    let (dimx, dimy) = image.dimensions();
    let mut neighbourhood = Vec::new();
    let mut coords = Vec::new();
    for xoffset in 0..width {
        for yoffset in 0..width {
            let xj = x + xoffset - width / 2;
            let yj = y + yoffset - width / 2;
            coords.push(Coord::new(yj, xj));
            //check if the coordinate is valid
            neighbourhood.push({
                if xj >= dimx || yj >= dimy {
                    image::Rgba([0, 0, 0, 255])
                } else {
                    image::Rgba(image.get_pixel(xj, yj).data)
                }
            });
        }
    }
    TextureNeighbourhood::new(Coord::new(x, y), coords, neighbourhood)
}

pub fn build_kdtree(
    neighbourhoods: &[TextureNeighbourhood],
) -> KdTree<f64, usize, std::vec::Vec<f64>> {
    let tree_dim = neighbourhoods[0].to_float64().len();
    let mut kdtree = KdTree::new(tree_dim);

    for (i, n) in neighbourhoods.iter().enumerate() {
        kdtree.add(n.to_float64(), i).unwrap();
    }

    return kdtree;
}

pub fn get_closest_neighbourhood_index(
    kd_tree: &KdTree<f64, usize, std::vec::Vec<f64>>,
    neighbourhood: &TextureNeighbourhood,
) -> usize {
    let m = kd_tree
        .nearest(&neighbourhood.to_float64(), 1, &squared_euclidean)
        .unwrap();
    return *m[0].1;
}

pub fn discrete_cosine_transform(pattern: &mut Vec<f64>, levels: usize) -> Vec<f64> {
    let mut planner = DCTplanner::new();
    let dct = planner.plan_dct2(pattern.len() / 3);

    let mut pattern_r: Vec<f64> = Vec::new(); //Vec::from_iter(pattern.iter().step_by(3));
    let mut pattern_g: Vec<f64> = Vec::new(); // Vec::from_iter(pattern[1..].iter().step_by(3));
    let mut pattern_b: Vec<f64> = Vec::new(); // Vec::from_iter(pattern[2..].iter().step_by(3));

    for (i, color) in pattern.iter().enumerate() {
        match i.clone() as u8 % 3 {
            0 => pattern_r.push(color.clone()),
            1 => pattern_g.push(color.clone()),
            _ => pattern_b.push(color.clone()),
        }
    }

    let mut output_r = vec![0f64; pattern.len() / 3];
    let mut output_g = vec![0f64; pattern.len() / 3];
    let mut output_b = vec![0f64; pattern.len() / 3];

    dct.process_dct2(&mut pattern_r, &mut output_r);
    dct.process_dct2(&mut pattern_g, &mut output_g);
    dct.process_dct2(&mut pattern_b, &mut output_b);

    output_r = output_r[0..levels / 3].to_vec();
    output_g = output_g[0..levels / 3].to_vec();
    output_b = output_b[0..levels / 3].to_vec();

    output_r.append(&mut output_g);
    output_r.append(&mut output_b);

    output_r
}

pub fn init_random_image(size: (u32, u32), seed: u32) -> image::RgbaImage {
    let mut image = image::RgbaImage::new(size.0, size.1);
    for x in 0..size.0 {
        for y in 0..size.1 {
            let r = Pcg32::seed_from_u64((seed + x * x - y) as u64).gen_range(0, 255);
            let g = Pcg32::seed_from_u64((seed + x * y + y * y) as u64).gen_range(0, 255);
            let b = Pcg32::seed_from_u64((seed * y + x * x) as u64).gen_range(0, 255);
            image[(x, y)] = image::Rgba([r, g, b, 255]);
        }
    }
    image
}
