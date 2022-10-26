import * as React from 'react'
import './App.css';
import Navbar from './components/Navbar';
import Template from './components/template';
import Masthead from './components/Masthead';
import { BrowserRouter as Router } from 'react-router-dom'

function App() {
  return (
    <Router>
      <Navbar/>
      <Masthead/>
      <Template/>
    </Router>
  );
}

export default App;
