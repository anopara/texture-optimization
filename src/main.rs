mod utils;
use crate::utils::*;
mod generator;
use crate::generator::*;
use image::GenericImage;
use kdtree::distance::squared_euclidean;
use std::path::Path;

use std::fs::{self, DirEntry};
use std::io;

//TODO list
//Tom's suggestion: shift centers of optimization neighbourhoods between iterations
// apply Discrete Cosine Transform
// TODO debug gaussian distance -> does it even look correct?

fn main() {
    //user parameters
    let neigh_width: Vec<usize> = vec![15, 25, 25, 35]; //
    let output_size = (150, 150);
    let input_size = (120, 120);
    let seed = 552436;
    let iters = 50;
    let resolution_levels = 4;

    let paths = fs::read_dir("imgs3/").unwrap();
    let mut imgs_paths = Vec::new();
    for path in paths {
        //println!("Name: {}", path.unwrap().path().display());
        imgs_paths.push(
            path.unwrap()
                .path()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
        );
    }

    let mut folder_number = 115;
    for img_path in imgs_paths.iter() {
        println!("processing image {}", img_path);
        let save_path = format!("out/{}/", folder_number);
        let full_img_path = format!("imgs3/{}", img_path);
        texture_optimization(
            full_img_path.as_str(),
            save_path.as_str(),
            &neigh_width,
            output_size,
            input_size,
            seed,
            iters,
            resolution_levels,
        );
        folder_number += 1;
    }
}

fn texture_optimization(
    example_image_path: &str,
    save_path: &str,
    neighbourhood_width: &Vec<usize>,
    output_size: (u32, u32),
    input_size: (u32, u32),
    initialization_seed: u32,
    iterations_num: usize,
    resolution_levels: usize,
) {
    let example_levels = init_resolution_levels(
        resolution_levels,
        input_size,
        (
            neighbourhood_width[resolution_levels - 1] as u32,
            neighbourhood_width[resolution_levels - 1] as u32,
        ),
    );
    let output_levels = init_resolution_levels(
        resolution_levels,
        output_size,
        (
            neighbourhood_width[resolution_levels - 1] as u32,
            neighbourhood_width[resolution_levels - 1] as u32,
        ),
    );

    println!("{:?} {:?}", example_levels, output_levels);

    for (i, lvl_res) in output_levels.iter().enumerate() {
        println!(
            "level: {} with out-resolution {:?} as in-resolution {:?}",
            i, lvl_res, example_levels[i as usize]
        );
        //1. initialize our output
        println!("initializing output");
        let mut output = match i {
            0 => init_random_image(*lvl_res, initialization_seed),
            _ => load_example_image(format!("{}final.png", save_path).as_str(), *lvl_res).to_rgba(),
        };
        //2.prepare example image
        println!("preparing example image");
        let example_image = load_example_image(example_image_path, example_levels[i]).to_rgba();

        let example_neighbourhoods =
            break_image_into_neighbourhoods(&example_image, neighbourhood_width[i], 1);
        /*
        example_neighbourhoods.clone_from_slice(&break_image_into_neighbourhoods(
            &image::imageops::flip_horizontal(&example_image),
            neighbourhood_width,
        ));
        example_neighbourhoods.clone_from_slice(&break_image_into_neighbourhoods(
            &image::imageops::flip_vertical(&example_image),
            neighbourhood_width,
        ));
        */

        let kd_tree = build_kdtree(&example_neighbourhoods);

        let update_rate = ((i + 1) as f64 / output_levels.len() as f64).max(0.1);
        println!("update rate {}", update_rate);
        let mut stop = false;

        //3.prepare targets map
        println!("preparing target map");
        let mut targets_map = TargetsMap::new(*lvl_res);
        //4.main resolve loop
        println!("Entering main resolve loop");
        for iter in 0..iterations_num {
            let prev_output = output.clone();
            if stop {
                break;
            }
            println!("iteration {}", iter);
            let output_neighbourhoods = break_image_into_neighbourhoods(
                &output,
                neighbourhood_width[i],
                (neighbourhood_width[i] / 4) as u32,
            );

            let iter_neighbourhood_width = neighbourhood_width[i];
            for neigh in output_neighbourhoods.iter() {
                //find closest
                let index = get_closest_neighbourhood_index(&kd_tree, &neigh);
                let closest_example_neighbourhood = &example_neighbourhoods[index];
                let weight_neighbourhood =
                    neighbourhood_weight(&neigh, &closest_example_neighbourhood, 0.8);

                //record per pixel targets
                for (i, example_pixel) in closest_example_neighbourhood.data.iter().enumerate() {
                    if check_coord_validity(&neigh.coords[i], *lvl_res) {
                        let weight_pixel = Coord::distanceGaussian(
                            &neigh.coords[i],
                            &neigh.center,
                            iter_neighbourhood_width as f64,
                        );
                        let coord_flat = neigh.coords[i].to_flat(*lvl_res);
                        //update targets map
                        targets_map.update(
                            coord_flat,
                            example_pixel,
                            weight_neighbourhood * weight_pixel,
                        );
                    }
                }
            }
            //energy minimize
            energy_minimization(&mut output, &targets_map, update_rate);
            targets_map.clear();
            //save image
            let path = format!("{}{}_{}.png", save_path, i, iter);
            let resize_output = image::imageops::resize(
                &output,
                output_size.0,
                output_size.1,
                image::imageops::Gaussian,
            );
            save_image_to_file(Path::new(path.as_str()), &resize_output);
            save_image_to_file(
                Path::new(format!("{}final.png", save_path).as_str()),
                &output,
            );

            println!("{}", image_dist(&prev_output, &output));
            if image_dist(&prev_output, &output) < 10000.0 {
                stop = true;
            }
        }
    }
}

fn init_resolution_levels(
    levels: usize,
    final_resolution: (u32, u32),
    min_img: (u32, u32),
) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let base: u32 = 2;
    for i in 0..levels {
        let x = final_resolution.0 / (base.pow((levels - i - 1) as u32));
        let y = final_resolution.1 / (base.pow((levels - i - 1) as u32));
        out.push((x.max(min_img.0), y.max(min_img.0)));
    }
    out
}

pub fn to_flat(x: u32, y: u32, dims: (u32, u32)) -> u32 {
    dims.0 * y + x
}

fn check_coord_validity(coord: &Coord, dims: (u32, u32)) -> bool {
    coord.x < dims.0 && coord.y < dims.1
}

fn neighbourhood_weight(n1: &TextureNeighbourhood, n2: &TextureNeighbourhood, r: f64) -> f64 {
    let mut dist = 0.0;
    for i in 0..n1.data.len() {
        for j in 0..3 {
            dist += ((n1.data[i][j] as f64) - (n2.data[i][j] as f64)).abs();
        }
    }
    dist.powf(r - 2.0)
}

fn image_dist(n1: &image::RgbaImage, n2: &image::RgbaImage) -> f64 {
    let (dimx, dimy) = n1.dimensions();
    let mut dist = 0.0;
    for x in 0..dimx {
        for y in 0..dimy {
            for i in 0..3 {
                dist += (((n1[(x, y)][i] - n2[(x, y)][i]) * (n1[(x, y)][i] - n2[(x, y)][i]))
                    as f64)
                    .sqrt();
            }
        }
    }
    dist
}

//stores all the targets and weights for pixels values after energy minimization
struct TargetsMap(Vec<PixelData>);

impl TargetsMap {
    fn new(output_size: (u32, u32)) -> Self {
        let mut target_map = Vec::new();
        for x in 0..output_size.0 {
            for y in 0..output_size.1 {
                target_map.push(PixelData::new(x, y));
            }
        }
        TargetsMap(target_map)
    }

    fn update(&mut self, coord_flat: u32, target: &image::Rgba<u8>, weight: f64) {
        let pixel_to_update = &mut self.0[coord_flat as usize];
        pixel_to_update.targets.push(target.clone());
        pixel_to_update.weights.push(weight);
    }

    fn clear(&mut self) {
        for pixel in self.0.iter_mut() {
            pixel.targets = Vec::new();
            pixel.weights = Vec::new();
        }
    }
}

#[derive(Clone)]
struct PixelData {
    coord: Coord,
    targets: Vec<image::Rgba<u8>>,
    weights: Vec<f64>,
}

impl PixelData {
    fn new(x: u32, y: u32) -> Self {
        PixelData {
            coord: Coord::new(x, y),
            targets: Vec::new(),
            weights: Vec::new(),
        }
    }

    //derivative  of the energey function doesnt depend on other pixels and is basically an average of all targets
    fn energy_minimize(
        &self,
        original_color: &image::Rgba<u8>,
        update_rate: f64,
    ) -> image::Rgba<u8> {
        let mut final_color = [0, 0, 0, 255];
        let max_weight: f64 = self.weights.iter().sum();
        for (i, t) in self.targets.iter().enumerate() {
            let normalized_weight = self.weights[i] / max_weight;
            final_color[0] += (normalized_weight * (t[0] as f64)) as u8;
            final_color[1] += (normalized_weight * (t[1] as f64)) as u8;
            final_color[2] += (normalized_weight * (t[2] as f64)) as u8;
        }
        //blend with original
        let final_color = [
            (final_color[0] as f64 * update_rate + original_color[0] as f64 * (1.0 - update_rate))
                as u8,
            (final_color[1] as f64 * update_rate + original_color[1] as f64 * (1.0 - update_rate))
                as u8,
            (final_color[2] as f64 * update_rate + original_color[2] as f64 * (1.0 - update_rate))
                as u8,
            255,
        ];
        image::Rgba(final_color)
    }
}

fn energy_minimization(image: &mut image::RgbaImage, targets_map: &TargetsMap, update_rate: f64) {
    for pixel in targets_map.0.iter() {
        let coord = (pixel.coord.x, pixel.coord.y);
        image[coord] = pixel.energy_minimize(&image[coord], update_rate);
    }
}
