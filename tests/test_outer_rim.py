"""The requested target is the maximum outer image rim; inner rings optional."""
import json
import cv2
import numpy as np
import pytest

from app.services.outer_rim import ALGORITHM, refine_outer_hole, select_outer
from test_refinement import api, synthetic


def test_one_outer_boundary_is_enough_without_calibration():
    image=np.full((500,700,3),190,np.uint8)
    cv2.ellipse(image,(350,250),(100,85),28,0,360,(30,30,30),-1,cv2.LINE_AA)
    box={'x1':245.,'y1':155.,'x2':455.,'y2':345.,'center_x':350.,'center_y':250.,'conf':.9}
    result=refine_outer_hole(image,box)
    assert result['fit_quality']['valid'],result['fit_quality']
    assert [result['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx([350,250],abs=.6)
    assert len(result['geometry']['outlines_px'])==1
    assert len(result['geometry']['edge_points_px'])==1


def test_inner_rim_position_cannot_change_selected_outer_image_center():
    image=np.full((500,700,3),190,np.uint8)
    box={'x1':240.,'y1':140.,'x2':460.,'y2':360.,'center_x':350.,'center_y':250.,'conf':.9}
    outputs=[]
    for inner_center in ((350,250),(380,220)):
        frame=image.copy();cv2.circle(frame,(350,250),100,(100,100,100),-1,cv2.LINE_AA)
        cv2.circle(frame,inner_center,45,(5,5,5),-1,cv2.LINE_AA)
        result=refine_outer_hole(frame,box)
        assert result['fit_quality']['valid'],result['fit_quality']
        outputs.append([result['refined_center'][k] for k in ('center_x','center_y')])
    np.testing.assert_allclose(outputs,[ [350,250],[350,250] ],atol=.6)


def test_smaller_low_residual_candidate_never_beats_outer_area():
    box=[0,0,200,200]
    candidates=[{'ellipse':((100,100),(160,160),0),'rms':.01},
                {'ellipse':((100,100),(180,180),0),'rms':1.},
                {'ellipse':((100,100),(100,100),0),'rms':.01}]
    selected,checks,reason=select_outer(candidates,box)
    assert selected[0]==1
    assert checks[2]['reason']=='too_small_for_outer_rim'


def test_larger_outline_outside_target_box_cannot_win():
    candidates=[{'ellipse':((100,100),(240,240),0),'rms':.01},
                {'ellipse':((100,100),(180,180),0),'rms':1.}]
    selected,checks,reason=select_outer(candidates,[0,0,200,200])
    assert selected[0]==1
    assert checks[0]['reason']=='outer_outline_outside_target_box'


def test_blank_or_only_small_inner_circle_has_no_outer_fallback():
    image=np.full((500,700,3),190,np.uint8)
    box={'x1':240.,'y1':140.,'x2':460.,'y2':360.,'center_x':350.,'center_y':250.,'conf':.9}
    assert refine_outer_hole(image,box)['refined_center'] is None
    cv2.circle(image,(350,250),50,(5,5,5),-1,cv2.LINE_AA)
    assert refine_outer_hole(image,box)['refined_center'] is None


@pytest.mark.parametrize('model',['usb_hole7','usb_hole11'])
def test_default_route_returns_one_outer_rim_without_k(api,monkeypatch,model):
    client,image,hole,truth,path=api
    monkeypatch.delenv('CAMERA_CALIBRATION_FILE',raising=False)
    status=client.get('/api/refinement/status').json()
    assert status['ready'] and status['calibration_required'] is False
    assert status['algorithm']==ALGORITHM
    _,encoded=cv2.imencode('.png',image)
    response=client.post('/api/'+model+'_refine_center',params={'target_u':hole['center_x'],'target_v':hole['center_y']},
                         files={'file':('frame.png',encoded.tobytes(),'image/png')})
    assert response.status_code==200
    data=response.json()['data'];target=data['target_hole']
    assert data['calibration_id'] is None
    assert data['center_definition']=='fitted_outer_ellipse_center'
    assert target['fit_quality']['valid'],target['fit_quality']
    assert target['refined_center']['reference']=='outer_rim'
    assert len(target['geometry']['outlines_px'])==1
    # Geometric ellipse center is independently estimated from the rendered rim.
    _,_,_,_,points,_=synthetic()
    expected=cv2.fitEllipse(points[0].astype('float32'))[0]
    assert [target['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx(expected,abs=.5)
    audit=json.loads((path/'audit/refinement.jsonl').read_text().splitlines()[-1])
    assert audit['calibration_snapshot'] is None
