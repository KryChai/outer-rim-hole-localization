"""Caliper regression fixtures: fragmented edges, spurious edges and rejection."""
from dataclasses import replace
import json
import time

import cv2
import numpy as np
import pytest

from app.services.caliper import (CaliperConfig, detect_boundaries, normal_samples,
                                  orthogonal_residual, ellipse_points)
from app.services.refinement import refine_hole
from test_refinement import synthetic


def box(hole):
    return [hole[k] for k in ('x1','y1','x2','y2')]


def test_calipers_keep_multiple_peaks_and_allow_missing_edges():
    cal,image,hole,*_=synthetic()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    seed=(((hole['x1']+hole['x2'])/2,(hole['y1']+hole['y2'])/2),
          (hole['x2']-hole['x1'],hole['y2']-hole['y1']),0.)
    cfg=CaliperConfig()
    samples=normal_samples(gray,seed,np.arange(-45,12,.5),cfg)
    assert len(samples['raw'])>cfg.calipers
    assert any(np.sum(samples['group']==g)>=2 for g in range(cfg.calipers))
    empty=normal_samples(np.full_like(gray,127),seed,np.arange(-45,12,.5),cfg)
    assert len(empty['raw'])==0


def test_orthogonal_distance_is_zero_on_rotated_ellipse():
    ellipse=((100.,120.),(60.,100.),37.)
    params=np.r_[ellipse[0],np.log(np.array(ellipse[1])/2),np.deg2rad(ellipse[2])]
    assert np.max(np.abs(orthogonal_residual(ellipse_points(ellipse),params)))<1e-8


def test_fixed_sampling_is_reproducible_and_diagnostics_json_serializable():
    cal,image,hole,*_=synthetic()
    first,diag=detect_boundaries(image,box(hole),cal)
    second,_=detect_boundaries(image,box(hole),cal)
    assert len(first)==len(second)==2
    np.testing.assert_array_equal(first[0]['edges'],second[0]['edges'])
    assert diag['accepted_candidates'][0]['selected_indices']
    assert len(diag['initial_signed_strength'])==diag['sample_count']
    json.dumps(diag,allow_nan=False)


@pytest.mark.parametrize('noise_seed',range(4))
def test_sensor_noise_does_not_move_physical_reference(noise_seed):
    cal,image,hole,truth,*_=synthetic()
    rng=np.random.default_rng(noise_seed)
    noisy=np.clip(image.astype(float)+rng.normal(0,2,image.shape),0,255).astype('uint8')
    fit=refine_hole(noisy,hole,cal)
    assert fit['fit_quality']['valid'],fit['fit_quality']
    assert [fit['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx(truth,abs=.5)


def test_scratches_breaking_connected_contours_still_recover_two_boundaries():
    cal,image,hole,truth,*_=synthetic()
    # Thin radial scratches connect and split multiple physical boundaries.
    center=np.array([hole['center_x'],hole['center_y']])
    for theta in np.deg2rad([12,83,149,215,282]):
        ray=np.array([np.cos(theta),np.sin(theta)])
        a,b=np.rint(center+ray*20).astype(int),np.rint(center+ray*85).astype(int)
        cv2.line(image,tuple(a),tuple(b),(220,220,220),1)
    fit=refine_hole(image,hole,cal)
    assert fit['fit_quality']['valid'],fit['fit_quality']
    assert [fit['refined_center'][k] for k in ('center_x','center_y')]==pytest.approx(truth,abs=.7)


def test_invalid_mask_excludes_overlays_without_inpainting():
    cal,image,hole,*_=synthetic()
    mask=np.zeros(image.shape[:2],np.uint8)
    mask[:,405:410]=1
    candidates,diag=detect_boundaries(image,box(hole),cal,invalid_mask=mask)
    assert len(candidates)>=2
    for candidate in candidates:
        raw=np.rint(candidate['edges']).astype(int)
        assert not mask[raw[:,1],raw[:,0]].any()


def test_missing_outer_cannot_relabel_two_deeper_circles_as_front():
    cal,image,hole,*_=synthetic()
    image[:]=210
    center=(int(hole['center_x']),int(hole['center_y']))
    cv2.circle(image,center,40,(70,70,70),-1,cv2.LINE_AA)
    cv2.circle(image,center,24,(10,10,10),-1,cv2.LINE_AA)
    fit=refine_hole(image,hole,cal)
    assert fit['refined_center'] is None
    assert fit['fit_quality']['reason']=='outer_reference_not_observed'


def test_blank_texture_and_large_occlusion_do_not_produce_control_centers():
    cal,image,hole,*_=synthetic()
    for variant in ('blank','texture','occluded'):
        bad=np.full_like(image,127)
        if variant=='texture':
            for x in range(0,image.shape[1],5):cv2.line(bad,(x,0),(x,image.shape[0]-1),(200,200,200),1)
        if variant=='occluded':
            bad=image.copy();bad[:,int(hole['center_x']):]=210
        fit=refine_hole(bad,hole,cal)
        assert fit['refined_center'] is None,variant
        assert fit['boundary_diagnostics'] is not None


def test_configuration_has_bounded_work_and_cannot_relax_rms_arbitrarily():
    cal,image,hole,*_=synthetic()
    for cfg in (replace(CaliperConfig(),max_rms_px=100),replace(CaliperConfig(),ransac_trials=100000)):
        with pytest.raises(ValueError,match='configuration'):detect_boundaries(image,box(hole),cal,cfg)


def test_original_resolution_large_hole_keeps_both_boundaries():
    image=np.full((1440,2560,3),180,np.uint8)
    cv2.circle(image,(1300,700),170,(70,70,70),-1,cv2.LINE_AA)
    cv2.circle(image,(1310,695),90,(10,10,10),-1,cv2.LINE_AA)
    candidates,diagnostics=detect_boundaries(image,[1120,520,1480,880])
    assert len(candidates)==2
    assert diagnostics['stage']=='candidates_ready'
    assert candidates[0]['ellipse'][0]==pytest.approx([1300,700],abs=.5)
    assert candidates[1]['ellipse'][0]==pytest.approx([1310,695],abs=.5)


def test_eccentric_deeper_mouth_is_not_lost_among_outer_ring_hypotheses():
    image=np.full((800,1000,3),180,np.uint8)
    cv2.ellipse(image,(500,400),(150,140),0,0,360,(90,90,90),-1,cv2.LINE_AA)
    cv2.ellipse(image,(550,350),(68,63),0,0,360,(8,8,8),-1,cv2.LINE_AA)
    candidates,diagnostics=detect_boundaries(image,[340,250,660,550])
    assert diagnostics['region_seed_count']>0
    assert any(np.linalg.norm(np.array(c['ellipse'][0])-[550,350])<1 for c in candidates)
    assert any(np.linalg.norm(np.array(c['ellipse'][0])-[500,400])<1 for c in candidates)
