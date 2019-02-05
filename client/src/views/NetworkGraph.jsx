import * as React from 'react';
import { Sigma, RelativeSize, RandomizeNodePositions, ForceAtlas2, EdgeShapes, NodeShapes, Graph } from 'react-sigma';
import axios from 'axios';
import classNames from "classnames";

// reactstrap components
import {
  Button
} from "reactstrap";

// let myGraph =
export class NetworkGraph extends React.Component<{}, Graph> {

  constructor(props: any) { // tslint:disable-line: no-any
    super(props);
    this.state = {
      nodes: [],
      edges: []
    }
  }

  updateSessionsGraphFull = () => {
    this.setState({
      nodes: [],
      edges: []
    })
    axios.get('http://app.margagl.io/api/sessions')
      .then(res => {
        let graph = {
          nodes: [],
          edges: []
        }
        let rmax = 1
        let rmin = 0
        let imax = 1
        let imin = 0
        for (let i = 0; i < res.data.data.length; i++) {
            const d = res.data.data[i]
            if (d.reward_mean !== null && d.reward_mean > rmax) {
                rmax = d.reward_mean
            }
            if (d.reward_mean !== null && d.reward_mean < rmin) {
                rmin = d.reward_mean
            }
            if (d.iteration > imax) {
                imax = d.iteration
            }
            if (d.iteration < imin) {
                imin = d.iteration
            }
        }
        rmax = rmax - rmin
        imax = imax - imin
        for (let i = 0; i < res.data.data.length; i++) {
          const d = res.data.data[i]
          if (d.reward_mean === null){
              continue
          }
          const node = {
              id: d.id,
              label: `${d.id} (${d.reward_mean})`,
              x: (d.iteration - imin)/imax,
              y: -(d.reward_mean - rmin)/rmax,
              color: '#00e3b4'
          }
          graph.nodes.push(node)
          if (d.parent_id !== null) {
            const edge = { id: d.id, source: d.parent_id, target: d.id, color: '#00e3b4' }
            graph.edges.push(edge)
          }
        }
        this.setState(graph)
        console.log(res.data.data)
        console.log(this.state)
    })
  }
  
  updateSessionsGraph = () => {
    this.setState({
      nodes: [],
      edges: []
    })
    axios.get('http://app.margagl.io/api/sessions?step_size=10')
      .then(res => {
        let graph = {
          nodes: [],
          edges: []
        }
        let rmax = 1
        let rmin = 0
        let imax = 1
        let imin = 0
        for (let i = 0; i < res.data.data.length; i++) {
            const d = res.data.data[i]
            if (d.reward_mean !== null && d.reward_mean > rmax) {
                rmax = d.reward_mean
            }
            if (d.reward_mean !== null && d.reward_mean < rmin) {
                rmin = d.reward_mean
            }
            if (d.iteration > imax) {
                imax = d.iteration
            }
            if (d.iteration < imin) {
                imin = d.iteration
            }
        }
        rmax = rmax - rmin
        imax = imax - imin
        let prev_d = null
        for (let i = 0; i < res.data.data.length; i++) {
          const d = res.data.data[i]
          if (d.reward_mean === null){
              continue
          }
          const node = {
              id: d.id,
              label: `${d.id} (${d.reward_mean})`,
              x: (d.iteration - imin)/imax,
              y: -(d.reward_mean - rmin)/rmax,
              size: (d.iteration - imin)*10/imax,
              color: '#00e3b4'
          }
          graph.nodes.push(node)
          if (prev_d !== null) {
            const edge = { id: d.id, source: prev_d, target: d.id, color: '#00e3b4' }
            graph.edges.push(edge)
          }
          prev_d = d.id
        }
        this.setState(graph)
        console.log(res.data.data)
        console.log(this.state)
    })
  }

  componentDidMount() {
    this.updateSessionsGraph()
  }

  render() {
      return (
        <div>
          <Button
            tag="label"
            className={classNames("btn-simple", "active")}
            color="info"
            id="0"
            size="sm"
            onClick={this.updateSessionsGraphFull}
          >
            <span className="d-none d-sm-block d-md-block d-lg-block d-xl-block">
              Update
            </span>
            <span className="d-block d-sm-none">
              <i className="tim-icons icon-single-02" />
            </span>
          </Button>
        {this.state.nodes.length > 0 ? (
          <Sigma
            renderer="canvas"
            style={{width: '100%', height: '400px'}}
            settings={{
              drawLabels: false,
              clone: false
            }}
            graph={this.state}
          >
            <EdgeShapes default="curvedArrow"/>
            <NodeShapes default="circle"/>
            <RelativeSize initialSize={25}/>
          </Sigma>) : ('')
        }
        </div>
    );
  }
}
