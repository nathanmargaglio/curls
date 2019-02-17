import React from "react";
// nodejs library that concatenates classes
import classNames from "classnames";
// react plugin used to create charts
import { Line, Bar } from "react-chartjs-2";

import {NetworkGraph} from './NetworkGraph'
import axios from 'axios';

// reactstrap components
import {
  Button,
  ButtonGroup,
  Card,
  CardHeader,
  CardBody,
  CardTitle,
  DropdownToggle,
  DropdownMenu,
  DropdownItem,
  UncontrolledDropdown,
  Label,
  FormGroup,
  Input,
  Table,
  Row,
  Col,
  UncontrolledTooltip
} from "reactstrap";

let options = {
  maintainAspectRatio: false,
  legend: {
    display: false
  },
  tooltips: {
    backgroundColor: "#f5f5f5",
    titleFontColor: "#333",
    bodyFontColor: "#666",
    bodySpacing: 4,
    xPadding: 12,
    mode: "nearest",
    intersect: 0,
    position: "nearest"
  },
  responsive: true,
  scales: {
    yAxes: [
      {
        barPercentage: 1.6,
        gridLines: {
          drawBorder: false,
          color: "rgba(29,140,248,0.0)",
          zeroLineColor: "transparent"
        },
        ticks: {
          suggestedMin: 60,
          suggestedMax: 125,
          padding: 20,
          fontColor: "#9a9a9a"
        }
      }
    ],
    xAxes: [
      {
        barPercentage: 1.6,
        gridLines: {
          drawBorder: false,
          color: "rgba(29,140,248,0.1)",
          zeroLineColor: "transparent"
        },
        ticks: {
          padding: 20,
          fontColor: "#9a9a9a"
        }
      }
    ]
  }
};

let labels = [
  "JAN",
  "FEB",
  "MAR",
  "APR",
  "MAY",
  "JUN",
  "JUL",
  "AUG",
  "SEP",
  "OCT",
  "NOV",
  "DEp"
]

let datasets = [
  {
    label: "My First dataset",
    fill: true,
    borderColor: "#1f8ef1",
    borderWidth: 2,
    borderDash: [],
    borderDashOffset: 0.0,
    pointBackgroundColor: "#1f8ef1",
    pointBorderColor: "rgba(255,255,255,0)",
    pointHoverBackgroundColor: "#1f8ef1",
    pointBorderWidth: 20,
    pointHoverRadius: 4,
    pointHoverBorderWidth: 15,
    pointRadius: 4,
    data: [100, 70, 90, 70, 85, 60, 75, 60, 90, 80, 110, 100]
  }
]

export class EpisodePlot extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      datasets: datasets,
      labels: labels,
      options: options
    };
  }
  
  updateData = async () => {
    let res = await axios.get('http://curls.margagl.io/api/sessions/' + String(this.props.activeNode) + '/episodes')
    console.log(res)
    let data = []
    let labels = []

    for (let r of res.data.data) {
      data.push(r.total_reward)
      labels.push(r.iteration)
    }
    
    let datasets = this.state.datasets
    datasets[0].data = data
    
    this.setState({
      datasets: datasets,
      labels: labels
    })
  }
  
  render() {
    return (
      <Row>
        <Col xs="12">
          <Card className="card-chart">
            <CardHeader>
              <Row>
                <Col className="text-left" sm="6" onClick={() => this.updateData()}>
                  <h5 className="card-category">Episodes</h5>
                  <CardTitle tag="h2">Total Reward per Episode</CardTitle>
                </Col>
                <Col sm="6">
                </Col>
              </Row>
            </CardHeader>
            <CardBody>
              <div className="chart-area">
                <Line
                  data={{
                      datasets: this.state.datasets,
                      labels: this.state.labels
                  }}
                  options={this.state.options}
                />
              </div>
            </CardBody>
          </Card>
        </Col>
      </Row>
    );
  }
}