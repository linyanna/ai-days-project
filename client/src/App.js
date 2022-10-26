import * as React from 'react'
import './App.css';
import Navbar from './components/Navbar';
import Template from './components/template';
import Masthead from './components/Masthead';
import Services from './components/Services';
import Protfolio from './components/Protfolio';
import Clients from './components/Clients';
import Contact from './components/Contact';
import { BrowserRouter as Router } from 'react-router-dom'

function App() {
  return (
    <Router>
      <Navbar/>
      <Masthead/>
      <Services/>
      <Protfolio/>
      <Clients/>
      <Contact/>
      <Template/>
    </Router>
  );
}

export default App;
