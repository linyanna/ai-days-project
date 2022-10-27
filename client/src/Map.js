import React , {useRef, useEffect, useInsertionEffect} from 'react'
import {loadModules} from "esri-loader"


function Map() {
    const MapE1 = useRef(null)
    useEffect(
        ()=>{
            let view;

        loadModules(["esri/views/MapView", "esri/WebMap", "esri/layers/GeoJSONLayer"], {
            cdd:true
        }).then(([MapView,WebMap, GeoJSONLayer])=>{
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

            const geojsonLayer = new GeoJSONLayer({
                url: "http://opendata.dc.gov/datasets/81a9d9885947483aa2088d81b20bfe66_5.geojson"
            })

            webmap.add(geojsonLayer)
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
