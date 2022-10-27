import * as React from 'react'
import './App.css';
import Navbar from './components/Navbar';
import Template from './components/template';
import Masthead from './components/Masthead';
import About from './components/About';
import Team from './components/Team';
import Services from './components/Services';
import Protfolio from './components/Protfolio';
import Clients from './components/Clients';
import Contact from './components/Contact';
import { BrowserRouter as Router } from 'react-router-dom'
import Footer from './components/Footer';
import Map from './Map.js'
function App() {
  return (
    <Router>
      <Navbar/>
      <Masthead/>    
      <Services/>
      <Protfolio/>
      <About/>
      <Team/>
      {/* <Clients/> */}
      <Contact/>
      <Map/>
      {/* <Footer/> */}
      {/* <Template/> */}
    </Router>
  );
}

export default App;
