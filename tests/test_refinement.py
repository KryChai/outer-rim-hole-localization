"""Known synthetic camera/geometry, never real camera calibration or weights."""
import importlib
import json
import sys
import types

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.refinement import (Calibration, build_refinement_data, conic,
                                     load_calibration, recover_coaxial, refine_hole)


def synthetic(scale=1, distortion=None, offset=(50,20,500)):
    calibration=Calibration.from_dict({'camera_matrix':[[900,0,320],[0,900,240],[0,0,1]],
        'dist_coeffs':distortion or [0]*5,'image_size':[640,480]})
    n=np.array([.25,-.3,.92]);n/=np.linalg.norm(n)
    a=np.cross(n,[0,1,0]);a/=np.linalg.norm(a);b=np.cross(n,a)
    c=np.array(offset,dtype=float)*scale
    image=np.full((480,640,3),210,np.uint8)
    contours=[]
    for radius,depth,color in ((40,0,60),(25,12,5)):
        t=np.linspace(0,2*np.pi,360,endpoint=False)
        xyz=c+depth*scale*n+radius*scale*(np.cos(t)[:,None]*a+np.sin(t)[:,None]*b)
        uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),calibration.matrix,calibration.distortion)
        uv=uv.reshape(-1,2);contours.append(uv)
        cv2.fillPoly(image,[np.rint(uv).astype(np.int32)],(color,)*3,lineType=cv2.LINE_AA)
    lo=contours[0].min(0)-3;hi=contours[0].max(0)+3
    hole={'conf':.95,'center_x':float((lo[0]+hi[0])/2),'center_y':float((lo[1]+hi[1])/2),
          'x1':float(lo[0]),'y1':float(lo[1]),'x2':float(hi[0]),'y2':float(hi[1])}
    truth,_=cv2.projectPoints(c[None],np.zeros(3),np.zeros(3),calibration.matrix,calibration.distortion)
    return calibration,image,hole,truth.reshape(2),contours,n


def test_geometry_recovers_front_and_back_centers_without_metric_sizes():
    cal,image,hole,truth,points,n=synthetic()
    qs=[conic(cv2.fitEllipse(p.astype(np.float32))) for p in points]
    centers,axis,score=recover_coaxial(*qs,cal)
    assert centers[0]==pytest.approx(truth,abs=.001)
    assert abs(axis@n)> .99999
    assert np.linalg.norm(centers[0]-centers[1])>2  # different planes, do not average


@pytest.mark.parametrize('scale',[.2,1,20])
def test_global_scale_does_not_change_pixel_center(scale):
    cal,image,hole,truth,_,_=synthetic(scale)
    fit=refine_hole(image,hole,cal)
    assert fit['fit_quality']['valid'],fit['fit_quality']
    result=fit['refined_center']
    assert [result['center_x'],result['center_y']]==pytest.approx(truth,abs=.4)
    assert fit['fit_quality']['metric_scale_known'] is False


def test_distortion_and_original_pixel_contract():
    cal,image,hole,truth,_,_=synthetic(distortion=[-.12,.04,.001,-.002,0])
    fit=refine_hole(image,hole,cal)
    assert fit['fit_quality']['valid'],fit['fit_quality']
    assert [fit['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx(truth,abs=.5)


def test_no_pair_has_no_refined_box_fallback():
    cal,image,hole,_,_,_=synthetic()
    fit=refine_hole(np.full_like(image,210),hole,cal)
    assert fit['refined_center'] is None
    assert not fit['fit_quality']['valid']
    assert fit['detection_center']['center_x']==hole['center_x']


def test_single_boundary_is_not_mistaken_for_a_coaxial_pair():
    cal,image,hole,_,points,_=synthetic()
    image[:]=210
    cv2.fillPoly(image,[np.rint(points[0]).astype(np.int32)],(30,30,30),lineType=cv2.LINE_AA)
    assert refine_hole(image,hole,cal)['refined_center'] is None


def test_extra_step_cannot_change_the_outer_front_reference():
    cal,image,hole,truth,points,n=synthetic()
    a=np.cross(n,[0,1,0]);a/=np.linalg.norm(a);b=np.cross(n,a)
    t=np.linspace(0,2*np.pi,360,endpoint=False)
    xyz=np.array([50.,20.,500.])+6*n+32*(np.cos(t)[:,None]*a+np.sin(t)[:,None]*b)
    middle,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),cal.matrix,cal.distortion)
    image[:]=210
    for curve,color in ((points[0],100),(middle.reshape(-1,2),50),(points[1],5)):
        cv2.fillPoly(image,[np.rint(curve).astype(np.int32)],(color,)*3,lineType=cv2.LINE_AA)
    fit=refine_hole(image,hole,cal)
    assert fit['fit_quality']['valid'],fit['fit_quality']
    assert [fit['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx(truth,abs=.4)


def test_three_near_field_holes_are_fitted_independently():
    image=np.full((480,640,3),210,np.uint8)
    holes=[]
    for x in (-100,0,100):
        cal,part,hole,truth,_,_=synthetic(offset=(x,0,600))
        image=np.minimum(image,part)
        holes.append(hole)
    result=build_refinement_data(image,holes,cal,(holes[1]['center_x'],holes[1]['center_y']),20)
    assert len(result['holes'])==3
    assert all(h['fit_quality']['valid'] for h in result['holes'])
    assert result['target_hole'] is result['holes'][1]


def test_resolution_and_border_are_rejected():
    cal,image,hole,_,_,_=synthetic()
    assert refine_hole(image[:200],hole,cal)['fit_quality']['reason']=='calibration_resolution_mismatch'
    assert refine_hole(image,{**hole,'x1':0},cal)['fit_quality']['reason']=='hole_at_image_border'


def test_target_is_not_replaced_by_a_different_successful_hole():
    cal,image,hole,_,_,_=synthetic()
    result=build_refinement_data(image,[hole],cal,(50,50),20)
    assert result['target_hole'] is None
    assert result['reason']=='target_not_found'
    ambiguous=build_refinement_data(image,[hole,{**hole,'center_x':hole['center_x']+1}],cal,
                                    (hole['center_x'],hole['center_y']),20)
    assert ambiguous['target_hole'] is None


def test_invalid_or_missing_calibration(monkeypatch,tmp_path):
    monkeypatch.delenv('CAMERA_CALIBRATION_FILE',raising=False)
    with pytest.raises(ValueError,match='required'): load_calibration()
    for bad in ({}, {'camera_matrix':np.eye(3).tolist(),'dist_coeffs':[0]*5,'image_size':[0,480]}):
        path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): load_calibration(path)


class Tensor:
    def __init__(self,x):self.x=np.asarray(x)
    def cpu(self):return self
    def item(self):return self.x.item()
    def numpy(self):return self.x


def fake_result(hole):
    box=types.SimpleNamespace(cls=Tensor(0),conf=Tensor(.95),
        xyxy=Tensor([[hole[k] for k in ('x1','y1','x2','y2')]]))
    return types.SimpleNamespace(boxes=[box])


@pytest.fixture
def api(monkeypatch,tmp_path):
    cal,image,hole,truth,_,_=synthetic()
    spec={'camera_matrix':cal.matrix.tolist(),'dist_coeffs':cal.distortion.tolist(),'image_size':[640,480]}
    path=tmp_path/'synthetic-only.json';path.write_text(json.dumps(spec))
    monkeypatch.setenv('CAMERA_CALIBRATION_FILE',str(path))
    monkeypatch.setenv('REFINEMENT_LOG_DIR',str(tmp_path/'audit'))
    manager=types.ModuleType('app.services.model_manager')
    class ModelManager:
        def load_all(self):pass
        async def predict_single_async(self,*args):return fake_result(hole)
    manager.ModelManager=ModelManager
    monkeypatch.setitem(sys.modules,'app.services.model_manager',manager)
    # No Ultralytics import, no camera and no real HTTP server/socket.
    app=importlib.import_module('app.main').create_app()
    with TestClient(app) as client:
        yield client,image,hole,truth,tmp_path


@pytest.mark.parametrize('model',['usb_hole7','usb_hole11'])
@pytest.mark.parametrize('extension',['.png','.jpg'])
def test_real_multipart_routes_and_audit(api,model,extension):
    client,image,hole,truth,path=api
    assert client.get('/api/refinement/status',params={'mode':'coaxial'}).json()['ready']
    assert client.get('/api/refinement/status',params={'mode':'coaxial'}).json()['boundary_method']=='normal_caliper_ransac_v1'
    assert client.get('/api/refinement/status',params={'mode':'coaxial'}).json()['boundary_revision']==2
    ok,encoded=cv2.imencode(extension,image);assert ok
    result=client.post('/api/'+model+'_refine_center',params={'mode':'coaxial','target_u':hole['center_x'],'target_v':hole['center_y']},
                       files={'file':('frame'+extension,encoded.tobytes(),'image/png' if extension=='.png' else 'image/jpeg')})
    assert result.status_code==200,result.text
    data=result.json()['data'];target=data['target_hole']
    assert target['fit_quality']['valid'],target['fit_quality']
    assert [target['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx(truth,abs=.4)
    lines=(path/'audit/refinement.jsonl').read_text().splitlines()
    assert json.loads(lines[-1])['target_hole']['geometry']==target['geometry']
    assert json.loads(lines[-1])['target_hole']['boundary_diagnostics']['accepted_candidates']


def test_missing_calibration_prevents_prediction(api,monkeypatch):
    client,*_=api
    monkeypatch.delenv('CAMERA_CALIBRATION_FILE')
    assert client.get('/api/refinement/status',params={'mode':'coaxial'}).status_code==503


def test_legacy_whole_ring_count_is_model_metadata_not_model_name(api,monkeypatch):
    client,image,*_=api
    from app.services import detection
    counts=[]
    monkeypatch.setattr(detection,'build_center_fit_data',lambda result,conf,n:counts.append(n) or {})
    _,encoded=cv2.imencode('.png',image)
    for model in ('usb_hole7','usb_hole11'):
        assert client.post('/api/'+model+'_center',files={'file':('frame.png',encoded.tobytes())}).status_code==200
    assert counts==[12,8]
