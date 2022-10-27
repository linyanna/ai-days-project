import React from 'react'
import '../css/styles.css'
import justin from '../images/justin.jpg'
import christine from '../images/christine.jpg'
import zoe from '../images/zoe.jpg'
import haohui from '../images/haohui.jpg'
import yanna from '../images/yanna.jpg'

const Team = () => {
  return (
    <div>
        <section className="page-section bg-light" id="team" style={{paddingTop: '3rem'}}>
            <div className="container">
            <div className="text-center">
                <h2 className="section-heading text-uppercase">Our Amazing Team</h2>
            </div>
            <div className="row">
                <div className="col-lg-4">
                <div className="team-member">
                    <a href="https://www.linkedin.com/in/justin-andrilenas-8a422b213/">
                    <img className="mx-auto rounded-circle" src={justin} alt="..."/></a>
                    <h4>Justin Andrilenas</h4>
                    <p className="text-muted">University of Florida</p>
                    </div>
                </div>
                <div className="col-lg-4">
                <div className="team-member">
                    <a href="https://www.linkedin.com/in/christine-lin-9a0155189/">
                        <img className="mx-auto rounded-circle" src={christine} alt="..." /></a>
                    <h4>Christine Lin</h4>
                    <p className="text-muted">University of Florida</p>
                </div>
                </div>
                <div className="col-lg-4">
                <div className="team-member">
                <a href="https://www.linkedin.com/in/zoe-brown-581b2724a/">
                    <img className="mx-auto rounded-circle" src={zoe} alt="..." /></a>
                    <h4>Zoe Brown</h4>
                    <p className="text-muted">University of Florida</p>
                </div>
                </div>
                <div className="col-lg-2"></div>
                <div className="col-lg-4">
                <div className="team-member">
                <a href="https://www.linkedin.com/in/haohui-bao/">
                    <img className="mx-auto rounded-circle" src={haohui} alt="..." /></a>
                    <h4>Haohui Bao</h4>
                    <p className="text-muted">University of Florida</p>
                </div>
                </div>
                <div className="col-lg-4">
                <div className="col-lg-2"></div>
                <div className="team-member">
                <a href="https://www.linkedin.com/in/yanna-lin/">
                    <img className="mx-auto rounded-circle" src={yanna} alt="..." /></a>
                    <h4>Yanna Lin</h4>
                    <p className="text-muted">University of Florida</p>
                </div>
                </div>
            </div>
            </div>
        </section>
    </div>
  )
}

export default Team