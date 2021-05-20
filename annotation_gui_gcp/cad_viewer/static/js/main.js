import * as THREE from 'https://unpkg.com/three/build/three.module.js';
import { FBXLoader } from 'https://unpkg.com/three/examples/jsm/loaders/FBXLoader.js';
import { OrbitControls } from 'https://unpkg.com/three/examples/jsm/controls/OrbitControls.js';

//----Variables----//
//DOM element to attach the renderer to
let viewport;

//built-in three.js _cameraControls will be attached to this
let _cameraControls;

//viewport size
let viewportWidth = 800;
let viewportHeight = 600;

//camera attributes
const view_angle = 45;
const aspect = viewportWidth / viewportHeight;

//----Constructors----//
const renderer = new THREE.WebGLRenderer({ antialias: true });
const _scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(view_angle, aspect);

//constructs an instance of a white light
renderer.setClearColor(0x05CB63); // Background color
const pointLight = new THREE.PointLight(0xFFFFFF);
const ambientLight = new THREE.AmbientLight( 0x404040 ); // soft white light

let _raycaster;

// List of three.js 3D objects depicting annotated GCPs
let _gcps = {};

// cad model
let _cad_model = null;
// path to currently-loaded cad model
let _path_model = null;

function getCompoundBoundingBox(object3D) {
    let box = null;
    object3D.traverse(function (obj3D) {
        let geometry = obj3D.geometry;
        if (geometry === undefined) return;
        geometry.computeBoundingBox();
        if (box === null) {
            box = geometry.boundingBox;
        } else {
            box.union(geometry.boundingBox);
        }
    });
    return box;
}

function fitCameraToSelection(camera, controls, selection, fitOffset = 1.2) {

    const box = new THREE.Box3();

    for (const object of selection) box.expandByObject(object);

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxSize = Math.max(size.x, size.y, size.z);
    const fitHeightDistance = maxSize / (2 * Math.atan(Math.PI * camera.fov / 360));
    const fitWidthDistance = fitHeightDistance / camera.aspect;
    const distance = fitOffset * Math.max(fitHeightDistance, fitWidthDistance);

    const direction = controls.target.clone()
        .sub(camera.position)
        .normalize()
        .multiplyScalar(distance);

    controls.maxDistance = distance * 10;
    controls.target.copy(center);
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.updateProjectionMatrix();
    camera.position.copy(controls.target).sub(direction);

    controls.update();

}


function load_cad_model(path_model) {
    if (path_model == _path_model) {
        return;
    }

    console.log("Loading CAD model " + path_model)
    const loader = new FBXLoader();
    loader.load(path_model, function (object) {
        _cad_model = object;
        _path_model = path_model;
        _scene.add(object);

        // Set the camera position to center of bbox
        const bbox = getCompoundBoundingBox(object)
        let cameraTarget = new THREE.Vector3();
        bbox.getCenter(cameraTarget);
        object.localToWorld(cameraTarget);

        _cameraControls.target = cameraTarget;
        fitCameraToSelection(camera, _cameraControls, object.children)
    });
}



function setup_scene() {
    //Sets up the renderer to the same size as a DOM element
    //and attaches it to that element
    renderer.setSize(viewportWidth, viewportHeight);
    viewport = document.getElementById('viewport');
    viewport.appendChild(renderer.domElement);


    _cameraControls = new OrbitControls(camera, renderer.domElement);
    _cameraControls.movementSpeed = 1;
    _cameraControls.domElement = viewport;


    _raycaster = new THREE.Raycaster();
    _raycaster.params.Points.threshold = 0.1;

    camera.position.set(10, 10, 10);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    pointLight.position.set(10, 50, 150);
    _scene.add(camera);
    _scene.add(pointLight);
    _scene.add(ambientLight);
}


function makeTextSprite(message, parameters) {
    if (parameters === undefined) parameters = {};

    var fontface = parameters.hasOwnProperty("fontface") ?
        parameters["fontface"] : "Arial";

    var fontsize = parameters.hasOwnProperty("fontsize") ?
        parameters["fontsize"] : 18;

    var borderThickness = parameters.hasOwnProperty("borderThickness") ?
        parameters["borderThickness"] : 4;

    var borderColor = parameters.hasOwnProperty("borderColor") ?
        parameters["borderColor"] : { r: 0, g: 0, b: 0, a: 1.0 };

    var backgroundColor = parameters.hasOwnProperty("backgroundColor") ?
        parameters["backgroundColor"] : { r: 255, g: 255, b: 255, a: 1.0 };

    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    context.font = "Bold " + fontsize + "px " + fontface;

    // get size data (height depends only on font size)
    var metrics = context.measureText(message);
    var textWidth = metrics.width;

    // background color
    context.fillStyle = "rgba(" + backgroundColor.r + "," + backgroundColor.g + ","
        + backgroundColor.b + "," + backgroundColor.a + ")";
    // border color
    context.strokeStyle = "rgba(" + borderColor.r + "," + borderColor.g + ","
        + borderColor.b + "," + borderColor.a + ")";

    context.lineWidth = borderThickness;
    roundRect(context, borderThickness / 2, borderThickness / 2, textWidth + borderThickness, fontsize * 1.4 + borderThickness, 6);
    // 1.4 is extra height factor for text below baseline: g,j,p,q.

    // text color
    context.fillStyle = "rgba(0, 0, 0, 1.0)";

    context.fillText(message, borderThickness, fontsize + borderThickness);

    // canvas contents will be used for a texture
    var texture = new THREE.Texture(canvas)
    texture.needsUpdate = true;

    var spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    var sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(500, 250, 1.0);
    return sprite;
}

// function for drawing rounded rectangles
function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
}

function update_text(data) {
    const txt = "Annotating point " + data.selected_point + " on file " + data.cad_filename;
    const header = document.getElementById("header");
    header.innerHTML = txt;
}

function updateGCPLabels(){
    for (var gcp_id in _gcps) {
        const sphere = _gcps[gcp_id]["marker"];
        const sprite = _gcps[gcp_id]["label"]
        const director_vector = new THREE.Vector3();
        director_vector.subVectors(camera.position, sphere.position);
        const cam_to_gcp_distance = director_vector.length();
        sprite.scale.set(cam_to_gcp_distance/5, cam_to_gcp_distance/10, 1.0);
        sprite.position.addVectors(sphere.position, director_vector.setLength(300.0));
        sprite.center.set(0,1)
    }
}

function update_gcps(annotations) {
    for (var gcp_id in _gcps) {
        if (!annotations.hasOwnProperty(gcp_id)){
            _scene.remove(_gcps[gcp_id]["marker"])
            _scene.remove(_gcps[gcp_id]["label"])
            delete _gcps[gcp_id];
        }
    }
    for (var gcp_id in annotations) {
        const xyz = annotations[gcp_id]["coordinates"];
        const gcp_position = new THREE.Vector3(xyz[0], xyz[1], xyz[2]);
        const color = annotations[gcp_id]["color"];
        let sphere;
        let sprite;
        // check if the property/key is defined in the object itself, not in parent
        if (!_gcps.hasOwnProperty(gcp_id)) {
            const sphereGeometry = new THREE.SphereGeometry(50);
            sphere = new THREE.Mesh(sphereGeometry);
            sprite = makeTextSprite(gcp_id, { fontsize: 32, fontface: "Georgia", borderColor: { r: color[0], g: color[1], b: color[2], a: 1.0 } });
            _scene.add(sphere);
            _scene.add(sprite);
            _gcps[gcp_id] = { "marker": sphere, "label": sprite };
        }
        else {
            sphere = _gcps[gcp_id]["marker"];
            sprite = _gcps[gcp_id]["label"]
        }
        sphere.position.copy(gcp_position);
        sphere.material.color = { 'r': color[0] / 255.0, 'g': color[1] / 255.0, 'b': color[2] / 255.0 };
        sphere.material.needsupdate = true;
    }
    updateGCPLabels();
}

function point_camera_at_xyz(point){
    _cameraControls.target.copy(point);

    // const direction = controls.target.clone()
    //     .sub(camera.position)
    //     .normalize()
    //     .multiplyScalar(distance);

    // camera.updateProjectionMatrix();
    // camera.position.copy(controls.target).sub(direction);

    _cameraControls.update();

}

function initialize_event_source() {
    let sse = new EventSource("/stream");
    sse.addEventListener("sync", function (e) {
        const data = JSON.parse(e.data)
        load_cad_model("/static/resources/cad_models/" + data.cad_filename);
        update_gcps(data.annotations);
        update_text(data);
    })
    sse.addEventListener("move_camera", function (e) {
        const data = JSON.parse(e.data);
        point_camera_at_xyz(data);
    })
}

function initialize() {
    setup_scene();

    // window.addEventListener('resize', onWindowResize, false);
    document.addEventListener('pointerdown', onDocumentMouseClick, false);

    // Event source for server-to-client pushed updates
    initialize_event_source();

    // call update
    update();
}

function post_json(data) {
    const method = 'POST';
    const headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    };
    const body = JSON.stringify(data, null, 4);
    const url = 'postdata';
    fetch(url, { method, headers, body })
}

function remove_point_observation() {
    const data = {
        command: "remove_point_observation",
    };
    post_json(data);
}

function add_or_update_point_observation(xyz) {
    const data = {
        xyz: xyz,
        command: "add_or_update_point_observation",
    };
    post_json(data);
}

function onDocumentMouseClick(event) {

    if (!event.ctrlKey) {
        return
    }

    event.preventDefault();

    switch (event.button) {
        case 0: // left
            const pickposition = setPickPosition(event)
            _raycaster.setFromCamera(pickposition, camera);

            const intersections = _raycaster.intersectObject(_cad_model, true);
            const intersection = (intersections.length) > 0 ? intersections[0] : null;

            if (intersection !== null) {
                const xyz = [intersection.point['x'], intersection.point['y'], intersection.point['z']];
                add_or_update_point_observation(xyz);
            }
            break;
        case 1: // middle
            break;
        case 2: // right
            remove_point_observation();
            break;
    }

}

function getCanvasRelativePosition(event) {
    const rect = viewport.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * window.innerWidth / rect.width,
        y: (event.clientY - rect.top) * window.innerHeight / rect.height,
    };
}

function setPickPosition(event) {
    const pos = getCanvasRelativePosition(event);
    return {
        x: (pos.x / window.innerWidth) * 2 - 1,
        y: (pos.y / window.innerHeight) * -2 + 1,  // note we flip Y
    };
}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

}

//a cross-browser method for efficient animation, more info at:
// http://paulirish.com/2011/requestanimationframe-for-smart-animating/
window.requestAnimFrame = (function () {
    return window.requestAnimationFrame ||
        window.webkitRequestAnimationFrame ||
        window.mozRequestAnimationFrame ||
        window.oRequestAnimationFrame ||
        window.msRequestAnimationFrame ||
        function (callback) {
            window.setTimeout(callback, 1000 / 60);
        };
})();

//----Update----//
function update() {
    // requests the browser to call update at it's own pace
    requestAnimFrame(update);

    // Update GCP labels so that they track the camera
    updateGCPLabels();

    _cameraControls.update(1);

    pointLight.position.copy( camera.position );

    draw();
}

function draw() {
    renderer.render(_scene, camera);
}

document.onload = initialize();
