import React , {useRef, useEffect, useInsertionEffect} from 'react'
import {loadModules} from "esri-loader"


function Map() {
    const MapE1 = useRef(null)
    useEffect(
        ()=>{
            let view;

        loadModules(["esri/views/MapView", "esri/WebMap", "esri/layers/CSVLayer"], {
            cdd:true
        }).then(([MapView,WebMap, CSVLayer])=>{
            const webmap = new WebMap({
                basemap: 'topo-vector'
            })

            view = new MapView({
                map:webmap,
                center:[-83,42],
                zoom:8,
                // use the ref as a container
                container:MapE1.current
            })

            const renderer = {
                type: "heatmap",
                colorStops: [
                  { color: "rgba(63, 40, 102, 0)", ratio: 0 },
                  { color: "#472b77", ratio: 0.083 },
                  { color: "#4e2d87", ratio: 0.166 },
                  { color: "#563098", ratio: 0.249 },
                  { color: "#5d32a8", ratio: 0.332 },
                  { color: "#6735be", ratio: 0.415 },
                  { color: "#7139d4", ratio: 0.498 },
                  { color: "#7b3ce9", ratio: 0.581 },
                  { color: "#853fff", ratio: 0.664 },
                  { color: "#a46fbf", ratio: 0.747 },
                  { color: "#c29f80", ratio: 0.83 },
                  { color: "#e0cf40", ratio: 0.913 },
                  { color: "#ffff00", ratio: 1 }
                ],
                maxDensity: 0.01,
                minDensity: 0,

                //referenceScale: 10000000,
              };

            const layer1 = new CSVLayer({
                url: "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.csv",
                renderer: renderer                
            })

            webmap.add(layer1)
        })
        
        return()=>{
            //close the map view
            if(!!view){
                view.destroy()
                view=null
            }
        }
        })
    return (
    <div style={{height:800}} ref={MapE1}>

    </div>
  )
}

export default Map
