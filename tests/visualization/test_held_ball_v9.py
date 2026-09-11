import unittest
import numpy as np
from mimicx.visualization import video_object_registration as registration


class HeldBallTests(unittest.TestCase):
    def setUp(self):
        self.track = {'sourceframe':[195,204,195], 'ball_world':[[1,2,3],[4,5,6],None]}
        self.config = {'held_source_intervals_inclusive':[[176,200]], 'attachment_body':'left_wrist_yaw_link',
                       'local_offset_m':[.1,0,0]}
        self.names=['left_wrist_yaw_link']
        self.pose=np.array([[10,20,30,1,0,0,0]],dtype=float)

    def test_held_follows_same_rigid_rule_without_mutating_raw(self):
        result=registration.display_ball_transform(self.track,0,self.names,self.pose,held_config=self.config)
        self.assertEqual(result['displayed_ball_world'],[10.1,20.,30.])
        self.assertEqual(self.track['ball_world'][0],[1,2,3])
        self.assertEqual(result['ball_role'],'held_left')

    def test_free_observation_keeps_reference_and_scene_translation(self):
        result=registration.display_ball_transform(self.track,1,self.names,self.pose,held_config=self.config,world_offset=[0,-7.5,.03])
        np.testing.assert_allclose(result['displayed_ball_world'],[4,-2.5,6.03])
        self.assertEqual(result['ball_role'],'source_registered')

    def test_null_never_becomes_held_ball(self):
        result=registration.display_ball_transform(self.track,2,self.names,self.pose,held_config=self.config)
        self.assertIsNone(result['displayed_ball_world'])
        self.assertEqual(result['ball_role'],'unobserved')

    def test_hand_rotation_applies_to_offset(self):
        self.pose[0,3:]=[np.sqrt(.5),0,0,np.sqrt(.5)]
        result=registration.display_ball_transform(self.track,0,self.names,self.pose,held_config=self.config)
        np.testing.assert_allclose(result['displayed_ball_world'],[10,20.1,30])

    def test_explicit_omission_preserves_observed_source_data(self):
        self.config['render_omission_source_frames']=[195]
        self.config['render_omission_reason']='depth or identity ambiguity'
        result=registration.display_ball_transform(self.track,0,self.names,self.pose,held_config=self.config)
        self.assertIsNone(result['displayed_ball_world'])
        self.assertEqual(result['ball_role'],'depth_or_identity_ambiguous')
        self.assertEqual(result['display_omission_reason'],'depth or identity ambiguity')
        self.assertEqual(self.track['ball_world'][0],[1,2,3])


if __name__=='__main__':unittest.main()
