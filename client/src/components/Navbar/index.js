import React from 'react'
import { Nav, NavbarContainer, NavLogo } from './NavbarElements'

const Navbar = () => {
  return (
    <>
      <Nav>
        <NavbarContainer>
          <NavLogo to="/">Navbar</NavLogo>
          <h1>AI Project</h1>
        </NavbarContainer>
      </Nav>
      
    </>
  )
}

export default Navbar