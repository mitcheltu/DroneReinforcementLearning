"use client";
import { Component, type ReactNode } from "react";

export default class SceneBoundary extends Component<{children:ReactNode},{failed:boolean}> {
  state={failed:false};
  static getDerivedStateFromError(){return {failed:true};}
  render(){return this.state.failed?<div className="canvas-message" role="alert">The 3D view is unavailable in this browser.<br />Try a browser with WebGL enabled. Playback telemetry is still available.</div>:this.props.children;}
}
