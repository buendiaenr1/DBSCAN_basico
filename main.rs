use ndarray::{Array2, Axis};
use csv::ReaderBuilder;
use petal_neighbors::distance::Euclidean;
use petal_clustering::{Dbscan, Fit};

use plotters::prelude::*;
use plotters::style::full_palette::*;

use plotters::prelude::{BitMapBackend, ChartBuilder, Circle, DrawingArea, IntoDrawingArea};
use plotters::style::{Color, WHITE};
use plotters::style::full_palette::{
    RED, GREEN, BLUE, CYAN, YELLOW, PURPLE, ORANGE, GREY,
};

use std::process::Command;

fn read_csv(file_path: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|field| field.parse::<f64>().unwrap()).collect();
        rows.push(row);
    }

    let num_rows = rows.len();
    let num_cols = rows[0].len();

    let flat_data: Vec<f64> = rows.into_iter().flatten().collect();

    Ok(Array2::from_shape_vec((num_rows, num_cols), flat_data)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
     // limpiar
     Command::new("cmd")
     .args(&["/C", "cls"])
     .status()
     .expect("Error al ejecutar el comando cls");
    println!("\n\n BUAP 2025   Enrique R.P. Buendia Lozada");

    // Leer datos desde CSV
    let points = read_csv("points.csv")?;

    // Ejecutar DBSCAN
    let clustering = Dbscan::new(3., 2, Euclidean::default()).fit(&points);

    println!("Número de clusters encontrados: {}", clustering.0.len());

    for (i, (label, indices)) in clustering.0.iter().enumerate() {
        println!("Cluster {} contiene los índices {:?}", label, indices);
        for &idx in indices {
            let point = points.row(idx);
            println!("Índice {}: [{:.1}, {:.1}]", idx, point[0], point[1]);
        }
    }

    println!("=====================================================");
    println!("Anomalías (ruido): {:?}", clustering.1);
    println!("=====================================================");

    // Calcular límites del gráfico
    let x_values = points.column(0).to_vec();
    let y_values = points.column(1).to_vec();

    let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 5.0;
    let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 5.0;
    let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 5.0;
    let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 5.0;

    // Crear backend de imagen
    let root = BitMapBackend::new("clusters.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Configurar gráfico
    let mut chart = ChartBuilder::on(&root)
        .caption("Clusters DBSCAN", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // Colores para los clusters
    let cluster_colors: Vec<RGBColor> = vec![
        RED,
        GREEN,
        BLUE,
        CYAN,
        YELLOW,
        PURPLE,
        ORANGE,
    ];

    // Dibujar puntos por cluster
    for (i, (label, indices)) in clustering.0.iter().enumerate() {
        let color = cluster_colors.get(i % cluster_colors.len()).copied();
        if let Some(color) = color {
            chart.draw_series(
                indices.iter().map(|&idx| {
                    let point = points.row(idx);
                    Circle::new((point[0], point[1]), 4, color.filled())
                })
            )?;
        }
    }

    // Dibujar anomalías (ruido) en gris
    chart.draw_series(
        clustering.1.iter().map(|&idx| {
            let point = points.row(idx);
            Circle::new((point[0], point[1]), 4, GREY.filled())
        })
    )?;

    root.present()?;
    println!("Gráfico guardado como 'clusters.png'");


    // Mantener la consola abierta después de salir
    Command::new("cmd")
            .args(&["/C", "cmd /k"])
            .status()
            .expect("Error al ejecutar el comando cmd /k");


    Ok(())
}