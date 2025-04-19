use ndarray::array;
use petal_neighbors::distance::Euclidean;
use petal_clustering::{Dbscan, Fit};


fn main (){
    let points = array![[1., 2.], [2., 2.], [2., 2.3], [8., 7.], [8., 8.], [25., 80.]];
    # parametros eps=3 radio de vecindad, minimo de puntos, metrica
    let clustering = Dbscan::new(3., 2, Euclidean::default()).fit(&points);

    println!(" Numero de clusters encontrados {}",clustering.0.len());
    
        println!(" puntos en el 0 cluster {:?}",clustering.0[&0]);
        println!(" puntos en el 1 cluster {:?}",clustering.0[&1]);
    
    println!(" Anomalias en los puntos {:?} ",clustering.1);

}