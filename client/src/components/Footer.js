import React from 'react'
import '../css/styles.css'
import footerimg from '../images/footershape.png';

const Footer = () => {
  return (
        <footer className="footer py-4" style={{  backgroundImage: "url(" + footerimg + ")", backgroundSize: 'cover', paddingTop: '90%'}}>
            <div className="container"  style={{marginTop: '4rem'}}>
            <div className="row align-items-center">
                <div className="col-lg-4 text-lg-start">Copyright © Your Website 2022</div>
                <div className="col-lg-4 my-3 my-lg-0">
                </div>
                <div className="col-lg-4 text-lg-end">
                <a className="link-dark text-decoration-none me-3" href="#!">Privacy Policy</a>
                <a className="link-dark text-decoration-none" href="#!">Terms of Use</a>
                </div>
            </div>
            </div>
        </footer>
  )
}

export default Footer